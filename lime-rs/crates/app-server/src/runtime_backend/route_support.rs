use super::{request_context::RuntimeModelSelection, ResolvedTurnRoute, RuntimeBackend};
use crate::{ExecutionRequest, RuntimeCoreError};
use app_server_protocol::{RouteFailure, RouteFailureCategory};
use lime_core::database::DbConnection;
use runtime_core::ModelRouteExclusion;
use serde_json::{json, Value};

pub(super) fn agent_control_route_snapshot_for_resolved_route(
    backend: &RuntimeBackend,
    route: &ResolvedTurnRoute,
    service_tier: Option<&str>,
) -> Result<Value, RuntimeCoreError> {
    let route_protocol = serde_json::to_value(&route.resolution.resolved_route.protocol)
        .map_err(|error| RuntimeCoreError::Backend(format!("serialize route protocol: {error}")))?;
    let provider_name = route
        .direct_provider_config
        .as_ref()
        .map(|config| config.provider_name.clone())
        .or_else(|| {
            backend
                .api_key_provider_service
                .get_provider(&route.db, &route.selection.provider)
                .ok()
                .flatten()
                .map(|provider| provider.provider.name)
        })
        .unwrap_or_else(|| route.selection.provider.clone());
    let auth = &route.resolution.resolved_route.auth;
    let direct_provider_config = route.direct_provider_config.as_ref();
    let model_registry = route
        .resolution
        .decision_payload
        .get("modelRegistry")
        .cloned()
        .unwrap_or(Value::Null);

    Ok(json!({
        "schemaVersion": 2,
        "routeSource": if route.direct_provider_config.is_some() {
            "direct_provider_config"
        } else {
            "catalog"
        },
        "providerPreference": route.selection.provider,
        "modelPreference": route.selection.model,
        "serviceTier": service_tier,
        "providerConfig": {
            "providerId": route.selection.provider,
            "providerName": provider_name,
            "modelName": route.selection.model,
            "reasoningEffort": route.selection.reasoning_effort,
            "toolshim": direct_provider_config.map(|config| config.toolshim),
            "toolshimModel": direct_provider_config
                .and_then(|config| config.toolshim_model.as_deref()),
            "supportsWebsockets": direct_provider_config
                .map(|config| config.supports_websockets)
        },
        "routeProtocol": route_protocol,
        "authKind": auth.kind,
        "credentialRef": auth.credential_ref,
        "effectiveGeneration": route.effective_generation,
        "modelRegistry": model_registry
    }))
}

pub(super) fn read_route_generation(db: &DbConnection) -> Result<u64, RuntimeCoreError> {
    let connection = db
        .lock()
        .map_err(|_| RuntimeCoreError::Backend("runtime database lock poisoned".to_string()))?;
    lime_core::database::dao::route_state::RouteStateDao::read_generation(&connection)
        .map_err(|error| RuntimeCoreError::Backend(format!("read route generation: {error}")))
}

pub(super) fn durable_credential_ref_for_generation<'a>(
    request: &'a ExecutionRequest,
    selection: &RuntimeModelSelection,
    generation: u64,
) -> Option<&'a str> {
    let route = request
        .runtime_request()?
        .metadata
        .as_ref()?
        .get("agentControlRoute")?;
    (route.get("schemaVersion").and_then(Value::as_u64) == Some(2)
        && route.get("effectiveGeneration").and_then(Value::as_u64) == Some(generation)
        && route.get("providerPreference").and_then(Value::as_str)
            == Some(selection.provider.as_str())
        && route.get("modelPreference").and_then(Value::as_str) == Some(selection.model.as_str()))
    .then(|| route.get("credentialRef").and_then(Value::as_str))
    .flatten()
    .map(str::trim)
    .filter(|value| !value.is_empty())
}

pub(super) fn runtime_error_from_route_failure(
    session_id: &str,
    selection: &RuntimeModelSelection,
    failure: &RouteFailure,
) -> RuntimeCoreError {
    let pending_after_route_generation_change = match &failure.category {
        RouteFailureCategory::NoCandidate => {
            matches!(
                failure.reason_code.as_str(),
                "no_candidate" | "routing_no_candidate"
            )
        }
        RouteFailureCategory::CapabilityGap => false,
        RouteFailureCategory::ProviderNeedsSetup => {
            failure.reason_code == "provider_not_configured"
        }
        RouteFailureCategory::ProviderDisabled => failure.reason_code == "provider_disabled",
        RouteFailureCategory::MissingCredential => failure.reason_code == "missing_enabled_api_key",
        RouteFailureCategory::ModelUnavailable => matches!(
            failure.reason_code.as_str(),
            "model_registry_metadata_missing" | "provider_models_cache_missing_requested_model"
        ),
        RouteFailureCategory::UnsupportedProtocol
        | RouteFailureCategory::UnsupportedEndpoint
        | RouteFailureCategory::InternalError => false,
    };
    if pending_after_route_generation_change {
        return RuntimeCoreError::PendingRoute {
            session_id: session_id.to_string(),
            provider: failure
                .provider_id
                .clone()
                .or_else(|| Some(selection.provider.clone())),
            model: failure
                .model_id
                .clone()
                .or_else(|| Some(selection.model.clone())),
            reason_code: failure.reason_code.clone(),
        };
    }

    if matches!(
        &failure.category,
        RouteFailureCategory::CapabilityGap
            | RouteFailureCategory::UnsupportedProtocol
            | RouteFailureCategory::UnsupportedEndpoint
    ) {
        return RuntimeCoreError::RouteRejected {
            session_id: session_id.to_string(),
            provider: failure
                .provider_id
                .clone()
                .or_else(|| Some(selection.provider.clone())),
            model: failure
                .model_id
                .clone()
                .or_else(|| Some(selection.model.clone())),
            category: failure.category.clone(),
            reason_code: failure.reason_code.clone(),
        };
    }

    RuntimeCoreError::Backend(format!(
        "App Server runtime backend route resolution failed: category={:?}, reason={}, provider={:?}, model={:?}, capability_gap={:?}",
        failure.category,
        failure.reason_code,
        failure.provider_id,
        failure.model_id,
        failure.capability_gap,
    ))
}

pub(super) fn runtime_route_exclusion(
    selection: &RuntimeModelSelection,
    direct_request: bool,
    credential_ref: Option<&str>,
    error: &lime_agent::ReplyAttemptError,
) -> Option<ModelRouteExclusion> {
    if direct_request {
        return None;
    }
    let classification = error.classification()?;
    match credential_ref {
        Some(credential_ref) if error.is_credential_reroutable_provider_failure() => {
            Some(ModelRouteExclusion::for_credential(
                selection.provider.clone(),
                selection.model.clone(),
                credential_ref,
                classification,
            ))
        }
        None if error.is_reroutable_provider_failure() => Some(ModelRouteExclusion::new(
            selection.provider.clone(),
            selection.model.clone(),
            classification,
        )),
        _ => None,
    }
}
