use crate::model_route;
use crate::TaskErrorRecord;
use runtime_core::{CanonicalLlmEvent, FailureClassification, FinishReason};
use serde_json::{json, Value};

pub(crate) fn image_running_payload_patch(
    payload: &Value,
    executor_mode: &str,
    completed_count: usize,
    failed_count: usize,
    requested_count: usize,
) -> Value {
    media_payload_patch(
        payload,
        "image_generation",
        executor_mode,
        vec![message_started_event()],
        json!({
            "status": "running",
            "completedCount": completed_count,
            "failedCount": failed_count,
            "requestedCount": requested_count,
        }),
    )
}

pub(crate) fn image_completed_payload_patch(
    payload: &Value,
    executor_mode: &str,
    image_count: usize,
    failed_count: usize,
) -> Value {
    media_payload_patch(
        payload,
        "image_generation",
        executor_mode,
        vec![
            media_output_event("image", image_count),
            turn_completed_event(),
        ],
        json!({
            "status": if failed_count > 0 { "partial" } else { "succeeded" },
            "imageCount": image_count,
            "failedCount": failed_count,
        }),
    )
}

pub(crate) fn image_failed_payload_patch(payload: &Value, error: &TaskErrorRecord) -> Value {
    let route = model_route::resolved_model_route_from_payload(payload);
    let executor_mode = route
        .as_ref()
        .and_then(|route| {
            model_route::image_executor_mode_from_route_protocol(route.protocol.as_deref())
        })
        .or_else(|| string_field(payload, &["executor_mode", "executorMode"]))
        .unwrap_or("images_api");
    failed_payload_patch(payload, "image_generation", executor_mode, error)
}

pub(crate) fn video_running_payload_patch(payload: &Value) -> Value {
    media_payload_patch(
        payload,
        "video_generation",
        video_executor_mode(payload),
        vec![message_started_event()],
        json!({
            "status": "running",
        }),
    )
}

pub(crate) fn video_completed_payload_patch(payload: &Value, video_url: Option<&str>) -> Value {
    media_payload_patch(
        payload,
        "video_generation",
        video_executor_mode(payload),
        vec![
            text_delta_event("video_generation_completed"),
            turn_completed_event(),
        ],
        json!({
            "status": "succeeded",
            "videoUrl": video_url,
        }),
    )
}

pub(crate) fn video_failed_payload_patch(payload: &Value, error: &TaskErrorRecord) -> Value {
    failed_payload_patch(
        payload,
        "video_generation",
        video_executor_mode(payload),
        error,
    )
}

fn video_executor_mode(payload: &Value) -> &str {
    model_route::resolved_model_route_from_payload(payload)
        .as_ref()
        .and_then(|route| {
            model_route::video_executor_mode_from_route_protocol(route.protocol.as_deref())
        })
        .or_else(|| string_field(payload, &["executor_mode", "executorMode"]))
        .unwrap_or("fal_video_generation")
}

fn failed_payload_patch(
    payload: &Value,
    task_family: &str,
    executor_mode: &str,
    error: &TaskErrorRecord,
) -> Value {
    media_payload_patch(
        payload,
        task_family,
        executor_mode,
        vec![turn_failed_event(error)],
        json!({
            "status": "failed",
            "errorCode": error.code,
            "errorStage": error.stage,
            "retryable": error.retryable,
        }),
    )
}

fn media_payload_patch(
    payload: &Value,
    task_family: &str,
    executor_mode: &str,
    runtime_events: Vec<Value>,
    diagnostics: Value,
) -> Value {
    let route = model_route::resolved_model_route_from_payload(payload);
    let diagnostics = json!({
        "taskFamily": task_family,
        "executorMode": executor_mode,
        "providerId": route.as_ref().and_then(|route| route.provider_id.clone()),
        "modelId": route.as_ref().and_then(|route| route.model_id.clone()),
        "protocol": route.as_ref().and_then(|route| route.protocol.clone()),
        "transport": "provider_http",
        "credential": "not_embedded",
        "eventOwner": "media_runtime",
        "details": diagnostics,
    });

    json!({
        "llm_events": runtime_events.clone(),
        "llmEvents": runtime_events,
        "provider_diagnostics": diagnostics.clone(),
        "providerDiagnostics": diagnostics,
    })
}

fn message_started_event() -> Value {
    runtime_event_value(
        "message.created",
        json!({
            "role": "assistant",
            "backend": "media_runtime",
        }),
        CanonicalLlmEvent::StepStart { index: 0 },
    )
}

fn media_output_event(media_kind: &str, result_count: usize) -> Value {
    json!({
        "type": "runtime.event",
        "payload": {
            "kind": "media_task_output",
            "mediaKind": media_kind,
            "resultCount": result_count,
            "backend": "media_runtime",
        },
    })
}

fn text_delta_event(text: &str) -> Value {
    runtime_event_value(
        "message.delta",
        json!({
            "text": text,
            "content": [{ "type": "text", "text": text }],
            "part": { "type": "text", "text": text },
            "backend": "media_runtime",
        }),
        CanonicalLlmEvent::TextDelta {
            id: "media-task-text".to_string(),
            text: text.to_string(),
        },
    )
}

fn turn_completed_event() -> Value {
    runtime_event_value(
        "turn.completed",
        json!({ "backend": "media_runtime" }),
        CanonicalLlmEvent::Finish {
            reason: FinishReason::Stop,
            usage: None,
            response_id: None,
        },
    )
}

fn turn_failed_event(error: &TaskErrorRecord) -> Value {
    runtime_event_value(
        "turn.failed",
        json!({
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
            "backend": "media_runtime",
        }),
        CanonicalLlmEvent::ProviderError {
            message: error.message.clone(),
            classification: Some(FailureClassification::ProviderInternal),
            retryable: Some(error.retryable),
        },
    )
}

fn runtime_event_value(event_type: &str, mut payload: Value, event: CanonicalLlmEvent) -> Value {
    if let Some(payload) = payload.as_object_mut() {
        payload.insert(
            "runtimeEvent".to_string(),
            serde_json::to_value(event).expect("canonical LLM event serializes"),
        );
    }
    json!({ "type": event_type, "payload": payload })
}

fn string_field<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|item| !item.is_empty())
    })
}
