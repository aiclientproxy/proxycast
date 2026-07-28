use serde_json::{json, Value};

mod helpers;
#[cfg(test)]
mod tests;

use self::helpers::*;

const MEDIA_TASK_WORKER_EXECUTOR_KIND: &str = "media_task_worker";
const RESOLVED_ROUTE_ENDPOINT_SOURCE: &str = "resolved_route";
const MEDIA_TASK_WORKER_CREDENTIAL_OWNER: &str = "media_task_worker";
const SECRET_NOT_EMBEDDED_STATUS: &str = "not_embedded";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ResolvedModelRouteProjection {
    pub(crate) provider_id: Option<String>,
    pub(crate) model_id: Option<String>,
    pub(crate) protocol: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelRouteFailureProjection {
    pub(crate) code: String,
    pub(crate) message: String,
    pub(crate) retryable: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RoutePayloadPreflight {
    pub(crate) payload_patch: Option<Value>,
    pub(crate) failure: Option<ModelRouteFailureProjection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouteExecutionSpec {
    pub(crate) binding_key: &'static str,
    pub(crate) task_label: &'static str,
}

pub(crate) fn image_generation_execution_spec() -> RouteExecutionSpec {
    RouteExecutionSpec {
        binding_key: "mediaTaskArtifact/image/create",
        task_label: "图片生成",
    }
}

pub(crate) fn video_generation_execution_spec() -> RouteExecutionSpec {
    RouteExecutionSpec {
        binding_key: "mediaTaskArtifact/video/create",
        task_label: "视频生成",
    }
}

pub(crate) fn image_route_payload_preflight(payload: &Value) -> RoutePayloadPreflight {
    route_payload_preflight(
        payload,
        &image_generation_execution_spec(),
        supports_image_generation_route_protocol,
    )
}

pub(crate) fn video_route_payload_preflight(payload: &Value) -> RoutePayloadPreflight {
    route_payload_preflight(
        payload,
        &video_generation_execution_spec(),
        supports_video_generation_route_protocol,
    )
}

pub(crate) fn resolved_model_route_from_payload(
    payload: &Value,
) -> Option<ResolvedModelRouteProjection> {
    let route = resolved_route_value(payload)?;
    let model_ref = object_field(route, &["modelRef", "model_ref"]);
    let auth = object_field(route, &["auth"]);
    let provider = object_field(route, &["provider"]);
    let model = object_field(route, &["model"]);

    let provider_id = model_ref
        .and_then(|value| string_field(value, &["providerId", "provider_id"]))
        .or_else(|| auth.and_then(|value| string_field(value, &["providerId", "provider_id"])))
        .or_else(|| provider.and_then(|value| string_field(value, &["id"])));
    let model_id = model_ref
        .and_then(|value| string_field(value, &["modelId", "model_id"]))
        .or_else(|| model.and_then(|value| string_field(value, &["id"])));
    let protocol = string_field(route, &["protocol"]);

    if provider_id.is_none() && model_id.is_none() && protocol.is_none() {
        return None;
    }

    Some(ResolvedModelRouteProjection {
        provider_id,
        model_id,
        protocol,
    })
}

pub(crate) fn route_failure_from_payload(payload: &Value) -> Option<ModelRouteFailureProjection> {
    let failure = route_failure_value(payload);
    let code = failure
        .and_then(|value| string_field(value, &["reasonCode", "reason_code"]))
        .or_else(|| string_field(payload, &["failure_code", "failureCode"]))
        .or_else(|| failure.and_then(|value| string_field(value, &["category"])))?;
    let category = failure
        .and_then(|value| string_field(value, &["category"]))
        .or_else(|| string_field(payload, &["failure_category", "failureCategory"]));
    let capability_gap = failure
        .and_then(|value| string_field(value, &["capabilityGap", "capability_gap"]))
        .or_else(|| string_field(payload, &["capability_gap", "capabilityGap"]));
    let message = failure
        .and_then(|value| string_field(value, &["message"]))
        .unwrap_or_else(|| route_failure_message(&code, category.as_deref(), capability_gap));
    let retryable = failure
        .and_then(|value| bool_field(value, &["retryable"]))
        .unwrap_or(false);

    Some(ModelRouteFailureProjection {
        code,
        message,
        retryable,
    })
}

pub(crate) fn route_execution_patch_from_payload(
    payload: &Value,
    spec: &RouteExecutionSpec,
) -> Option<Value> {
    resolved_model_route_from_payload(payload)?;
    if route_failure_from_payload(payload).is_some() {
        return None;
    }
    if route_execution_value(payload).is_some() {
        return None;
    }

    let binding = build_route_execution_binding(payload, spec)?;
    Some(json!({
        "model_route_execution": binding.clone(),
        "modelRouteExecution": binding
    }))
}

pub(crate) fn route_execution_failure_from_payload(
    payload: &Value,
    spec: &RouteExecutionSpec,
) -> Option<ModelRouteFailureProjection> {
    let route = resolved_model_route_from_payload(payload)?;
    let Some(execution) = route_execution_value(payload) else {
        return Some(unsupported_route_execution_failure(
            spec,
            "缺少模型路由执行绑定",
        ));
    };

    let executor = object_field(execution, &["executor"]);
    let credential_resolver =
        object_field(execution, &["credentialResolver", "credential_resolver"]);
    let execution_route = object_field(execution, &["route"]);
    let executor_kind = executor.and_then(|value| string_field(value, &["kind"]));
    if normalized_token(executor_kind.as_deref()) != Some(MEDIA_TASK_WORKER_EXECUTOR_KIND.into()) {
        return Some(unsupported_route_execution_failure(
            spec,
            "执行器必须是媒体任务 worker",
        ));
    }

    let binding_key =
        executor.and_then(|value| string_field(value, &["bindingKey", "binding_key"]));
    if binding_key.as_deref() != Some(spec.binding_key) {
        return Some(unsupported_route_execution_failure(
            spec,
            "执行绑定与当前媒体任务不匹配",
        ));
    }

    let endpoint_source =
        executor.and_then(|value| string_field(value, &["endpointSource", "endpoint_source"]));
    if normalized_token(endpoint_source.as_deref()) != Some(RESOLVED_ROUTE_ENDPOINT_SOURCE.into()) {
        return Some(unsupported_route_execution_failure(
            spec,
            "执行 endpoint 必须来自 resolved route",
        ));
    }

    let credential_owner = credential_resolver.and_then(|value| string_field(value, &["owner"]));
    if normalized_token(credential_owner.as_deref())
        != Some(MEDIA_TASK_WORKER_CREDENTIAL_OWNER.into())
    {
        return Some(unsupported_route_execution_failure(
            spec,
            "凭证解析 owner 必须是媒体任务 worker",
        ));
    }

    let secret_status = credential_resolver
        .and_then(|value| string_field(value, &["secretMaterialStatus", "secret_material_status"]));
    if normalized_token(secret_status.as_deref()) != Some(SECRET_NOT_EMBEDDED_STATUS.into())
        || credential_resolver_has_embedded_secret(credential_resolver)
    {
        return Some(unsupported_route_execution_failure(
            spec,
            "执行绑定不能携带明文凭证",
        ));
    }

    if let Some(provider_id) =
        execution_route.and_then(|value| string_field(value, &["providerId", "provider_id"]))
    {
        if route.provider_id.as_deref() != Some(provider_id.as_str()) {
            return Some(unsupported_route_execution_failure(
                spec,
                "执行绑定 provider 与 resolved route 不一致",
            ));
        }
    }

    if let Some(model_id) =
        execution_route.and_then(|value| string_field(value, &["modelId", "model_id"]))
    {
        if route.model_id.as_deref() != Some(model_id.as_str()) {
            return Some(unsupported_route_execution_failure(
                spec,
                "执行绑定 model 与 resolved route 不一致",
            ));
        }
    }

    if let Some(protocol) = execution_route.and_then(|value| string_field(value, &["protocol"])) {
        if normalized_token(route.protocol.as_deref()) != normalized_token(Some(&protocol)) {
            return Some(unsupported_route_execution_failure(
                spec,
                "执行绑定 protocol 与 resolved route 不一致",
            ));
        }
    }

    None
}

fn route_payload_preflight(
    payload: &Value,
    spec: &RouteExecutionSpec,
    supports_protocol: fn(Option<&str>) -> bool,
) -> RoutePayloadPreflight {
    if let Some(failure) = route_failure_from_payload(payload) {
        return RoutePayloadPreflight {
            payload_patch: None,
            failure: Some(failure),
        };
    }

    let Some(route) = resolved_model_route_from_payload(payload) else {
        return RoutePayloadPreflight::default();
    };

    if !supports_protocol(route.protocol.as_deref()) {
        return RoutePayloadPreflight {
            payload_patch: None,
            failure: Some(unsupported_protocol_failure(
                route.protocol.as_deref(),
                spec.task_label,
            )),
        };
    }

    if route_execution_value(payload).is_none() {
        if let Some(payload_patch) = route_execution_patch_from_payload(payload, spec) {
            return RoutePayloadPreflight {
                payload_patch: Some(payload_patch),
                failure: None,
            };
        }
    }

    RoutePayloadPreflight {
        payload_patch: None,
        failure: route_execution_failure_from_payload(payload, spec),
    }
}

pub(crate) fn image_executor_mode_from_route_protocol(
    protocol: Option<&str>,
) -> Option<&'static str> {
    match normalized_token(protocol)?.as_str() {
        "openai_responses" | "codex_responses" => Some("responses_image_generation"),
        "gemini_generate_content" => Some("gemini_generate_content"),
        "dashscope_multimodal_generation" => Some("dashscope_images"),
        "openai_images" | "fal" => Some("images_api"),
        _ => None,
    }
}

pub(crate) fn supports_image_generation_route_protocol(protocol: Option<&str>) -> bool {
    matches!(
        normalized_token(protocol).as_deref(),
        None | Some(
            "openai_images"
                | "openai_responses"
                | "codex_responses"
                | "fal"
                | "gemini_generate_content"
                | "dashscope_multimodal_generation"
        )
    )
}

pub(crate) fn supports_video_generation_route_protocol(protocol: Option<&str>) -> bool {
    video_executor_mode_from_route_protocol(protocol).is_some()
}

pub(crate) fn video_executor_mode_from_route_protocol(
    protocol: Option<&str>,
) -> Option<&'static str> {
    match normalized_token(protocol)?.as_str() {
        "fal" => Some("fal_video_generation"),
        "xai_video" => Some("xai_video_generation"),
        _ => None,
    }
}
