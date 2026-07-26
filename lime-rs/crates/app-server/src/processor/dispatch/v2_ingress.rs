use app_server_protocol::error_codes;
use app_server_protocol::protocol::v2::{ClientRequest, Method};
use app_server_protocol::{JsonRpcError, JsonRpcRequest, RequestId};

pub(super) fn decode(request: &JsonRpcRequest) -> Result<Option<ClientRequest>, JsonRpcError> {
    if Method::parse(&request.method).is_none() {
        return Ok(None);
    }
    if request
        .params
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .is_some_and(|params| params.contains_key("sessionId") || params.contains_key("session_id"))
    {
        return Err(JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            "v2 requests must use threadId; sessionId is not a supported field",
        ));
    }

    let value = serde_json::to_value(request).map_err(|error| {
        JsonRpcError::new(
            error_codes::INVALID_REQUEST,
            format!("failed to encode v2 request: {error}"),
        )
    })?;
    serde_json::from_value(value).map(Some).map_err(|error| {
        JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            format!("invalid v2 request params: {error}"),
        )
    })
}

pub(super) fn into_parts(
    request: ClientRequest,
) -> Result<(RequestId, String, Option<serde_json::Value>), JsonRpcError> {
    match request {
        ClientRequest::ThreadStart { id, params } => parts(id, Method::ThreadStart, params),
        ClientRequest::ThreadFork { id, params } => parts(id, Method::ThreadFork, params),
        ClientRequest::ThreadResume { id, params } => parts(id, Method::ThreadResume, params),
        ClientRequest::ThreadRead { id, params } => parts(id, Method::ThreadRead, params),
        ClientRequest::ThreadList { id, params } => parts(id, Method::ThreadList, params),
        ClientRequest::ThreadLoadedList { id, params } => {
            parts(id, Method::ThreadLoadedList, params)
        }
        ClientRequest::ThreadUnsubscribe { id, params } => {
            parts(id, Method::ThreadUnsubscribe, params)
        }
        ClientRequest::ThreadIncrementElicitation { id, params } => {
            parts(id, Method::ThreadIncrementElicitation, params)
        }
        ClientRequest::ThreadDecrementElicitation { id, params } => {
            parts(id, Method::ThreadDecrementElicitation, params)
        }
        ClientRequest::ThreadArchive { id, params } => parts(id, Method::ThreadArchive, params),
        ClientRequest::ThreadDelete { id, params } => parts(id, Method::ThreadDelete, params),
        ClientRequest::ThreadUnarchive { id, params } => parts(id, Method::ThreadUnarchive, params),
        ClientRequest::ThreadSetName { id, params } => parts(id, Method::ThreadSetName, params),
        ClientRequest::ThreadMetadataUpdate { id, params } => {
            parts(id, Method::ThreadMetadataUpdate, params)
        }
        ClientRequest::ThreadCompactStart { id, params } => {
            parts(id, Method::ThreadCompactStart, params)
        }
        ClientRequest::ThreadTurnsList { id, params } => parts(id, Method::ThreadTurnsList, params),
        ClientRequest::ThreadItemsList { id, params } => parts(id, Method::ThreadItemsList, params),
        ClientRequest::ThreadInjectItems { id, params } => {
            parts(id, Method::ThreadInjectItems, params)
        }
        ClientRequest::ThreadSearch { id, params } => parts(id, Method::ThreadSearch, params),
        ClientRequest::ThreadSearchOccurrences { id, params } => {
            parts(id, Method::ThreadSearchOccurrences, params)
        }
        ClientRequest::ThreadGoalSet { id, params } => parts(id, Method::ThreadGoalSet, params),
        ClientRequest::ThreadGoalGet { id, params } => parts(id, Method::ThreadGoalGet, params),
        ClientRequest::ThreadGoalClear { id, params } => parts(id, Method::ThreadGoalClear, params),
        ClientRequest::ArtifactWrite { id, params } => parts(id, Method::ArtifactWrite, params),
        ClientRequest::ModelList { id, params } => parts(id, Method::ModelList, params),
        ClientRequest::ModelProviderCapabilitiesRead { id, params } => {
            parts(id, Method::ModelProviderCapabilitiesRead, params)
        }
        ClientRequest::ThreadSettingsUpdate { id, params } => {
            parts(id, Method::ThreadSettingsUpdate, params)
        }
        ClientRequest::ThreadMemoryModeSet { id, params } => {
            parts(id, Method::ThreadMemoryModeSet, params)
        }
        ClientRequest::ThreadShellCommand { id, params } => {
            parts(id, Method::ThreadShellCommand, params)
        }
        ClientRequest::ThreadApproveGuardianDeniedAction { id, params } => {
            parts(id, Method::ThreadApproveGuardianDeniedAction, params)
        }
        ClientRequest::ThreadBackgroundTerminalsClean { id, params } => {
            parts(id, Method::ThreadBackgroundTerminalsClean, params)
        }
        ClientRequest::ThreadBackgroundTerminalsList { id, params } => {
            parts(id, Method::ThreadBackgroundTerminalsList, params)
        }
        ClientRequest::ThreadBackgroundTerminalsTerminate { id, params } => {
            parts(id, Method::ThreadBackgroundTerminalsTerminate, params)
        }
        ClientRequest::TurnStart { id, params } => parts(id, Method::TurnStart, params),
        ClientRequest::TurnSteer { id, params } => parts(id, Method::TurnSteer, params),
        ClientRequest::TurnInterrupt { id, params } => parts(id, Method::TurnInterrupt, params),
    }
}

fn parts(
    id: RequestId,
    method: Method,
    params: impl serde::Serialize,
) -> Result<(RequestId, String, Option<serde_json::Value>), JsonRpcError> {
    let params = serde_json::to_value(params).map_err(|error| {
        JsonRpcError::new(
            error_codes::INVALID_REQUEST,
            format!("failed to lower v2 request params: {error}"),
        )
    })?;
    Ok((id, method.as_str().to_string(), Some(params)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::protocol::v2::{
        METHOD_THREAD_READ, METHOD_THREAD_RESUME, METHOD_TURN_INTERRUPT,
    };
    use app_server_protocol::RequestId;
    use serde_json::json;

    fn request(method: &str, params: serde_json::Value) -> JsonRpcRequest {
        JsonRpcRequest::new(RequestId::Integer(1), method, Some(params))
    }

    #[test]
    fn non_v2_requests_continue_to_the_existing_catalog() {
        let request = request("initialize", json!({}));

        assert!(decode(&request).expect("decode request").is_none());
    }

    #[test]
    fn every_v2_method_uses_typed_param_validation() {
        let request = request(METHOD_THREAD_READ, json!({ "threadId": 42 }));

        let error = decode(&request).expect_err("invalid v2 params must fail closed");
        assert_eq!(error.code, error_codes::INVALID_PARAMS);
    }

    #[test]
    fn legacy_resume_shape_is_rejected_before_the_v0_handler() {
        let request = request(METHOD_THREAD_RESUME, json!({ "sessionId": "session-1" }));

        let error = decode(&request).expect_err("legacy resume params must fail closed");
        assert_eq!(error.code, error_codes::INVALID_PARAMS);
    }

    #[test]
    fn typed_resume_lowers_to_the_v2_dispatch_method() {
        let request = request(METHOD_THREAD_RESUME, json!({ "threadId": "thread-1" }));
        let request = decode(&request)
            .expect("decode request")
            .expect("v2 request");

        let (_, method, _) = into_parts(request).expect("lower request");
        assert_eq!(method, METHOD_THREAD_RESUME);
    }

    #[test]
    fn typed_request_lowers_to_dispatch_parts_without_v0_decode() {
        let request = request(
            METHOD_THREAD_READ,
            json!({
                "threadId": "thread-1",
                "includeTurns": true,
            }),
        );
        let request = decode(&request)
            .expect("decode request")
            .expect("v2 request");

        let (id, method, params) = into_parts(request).expect("lower request");
        assert_eq!(id, RequestId::Integer(1));
        assert_eq!(method, METHOD_THREAD_READ);
        assert_eq!(params.expect("params")["threadId"], "thread-1");
    }

    #[test]
    fn implemented_v2_methods_continue_to_the_current_handlers() {
        let request = request(
            METHOD_TURN_INTERRUPT,
            json!({ "threadId": "thread-1", "turnId": "turn-1" }),
        );
        let request = decode(&request)
            .expect("decode request")
            .expect("v2 request");

        let (_, method, _) = into_parts(request).expect("lower request");
        assert_eq!(method, METHOD_TURN_INTERRUPT);
    }
}
