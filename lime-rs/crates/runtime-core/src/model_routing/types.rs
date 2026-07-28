use serde_json::{json, Value};
use std::fmt;

use crate::FailureClassification;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeModelSelection {
    pub provider: String,
    pub model: String,
    pub source: &'static str,
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileModelSlot {
    pub slot: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub source: String,
    pub decision_reason: Option<String>,
    pub capability_tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRoutingDecision {
    pub service_model_slot: String,
    pub requested_provider: Option<String>,
    pub requested_model: Option<String>,
    pub settings_source: String,
    pub decision_reason: String,
    pub fallback_chain: Vec<String>,
    pub profile_slots: Vec<ProfileModelSlot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderReadiness {
    pub ready: bool,
    pub status: &'static str,
    pub source: &'static str,
    pub reason_code: Option<&'static str>,
    pub provider_type: Option<String>,
    pub enabled: Option<bool>,
    pub enabled_key_count: Option<usize>,
    pub total_key_count: Option<usize>,
    pub direct_request_config: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingResolution {
    pub selection: RuntimeModelSelection,
    pub routing: ModelRoutingDecision,
    pub readiness: ProviderReadiness,
    pub attempted: Vec<RoutingAttempt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingAttempt {
    pub slot: String,
    pub provider: String,
    pub model: String,
    pub source: String,
    pub readiness: ProviderReadiness,
    pub runtime_failure: Option<ModelRouteExclusion>,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ModelRouteExclusion {
    pub provider: String,
    pub model: String,
    pub reason_code: &'static str,
    pub classification: FailureClassification,
    pub retryable: bool,
    credential_ref: Option<String>,
}

impl ProviderReadiness {
    pub fn direct_request_ready() -> Self {
        Self {
            ready: true,
            status: "ready",
            source: "direct_provider_config",
            reason_code: None,
            provider_type: None,
            enabled: None,
            enabled_key_count: None,
            total_key_count: None,
            direct_request_config: true,
        }
    }

    pub fn direct_request_blocked(reason_code: &'static str) -> Self {
        Self {
            ready: false,
            status: "blocked",
            source: "direct_provider_config",
            reason_code: Some(reason_code),
            provider_type: None,
            enabled: None,
            enabled_key_count: None,
            total_key_count: None,
            direct_request_config: true,
        }
    }

    pub fn provider_not_configured() -> Self {
        Self {
            ready: false,
            status: "needs_setup",
            source: "provider_store",
            reason_code: Some("provider_not_configured"),
            provider_type: None,
            enabled: None,
            enabled_key_count: Some(0),
            total_key_count: Some(0),
            direct_request_config: false,
        }
    }

    pub fn runtime_failure(reason_code: &'static str) -> Self {
        Self {
            ready: false,
            status: "excluded",
            source: "runtime_failure",
            reason_code: Some(reason_code),
            provider_type: None,
            enabled: None,
            enabled_key_count: None,
            total_key_count: None,
            direct_request_config: false,
        }
    }

    pub fn provider_store_blocked(
        reason_code: &'static str,
        provider_type: Option<String>,
        enabled: Option<bool>,
        enabled_key_count: usize,
        total_key_count: usize,
    ) -> Self {
        Self {
            ready: false,
            status: "blocked",
            source: "provider_store",
            reason_code: Some(reason_code),
            provider_type,
            enabled,
            enabled_key_count: Some(enabled_key_count),
            total_key_count: Some(total_key_count),
            direct_request_config: false,
        }
    }

    pub fn provider_store_needs_setup(
        reason_code: &'static str,
        provider_type: Option<String>,
        enabled: Option<bool>,
        enabled_key_count: usize,
        total_key_count: usize,
    ) -> Self {
        Self {
            ready: false,
            status: "needs_setup",
            source: "provider_store",
            reason_code: Some(reason_code),
            provider_type,
            enabled,
            enabled_key_count: Some(enabled_key_count),
            total_key_count: Some(total_key_count),
            direct_request_config: false,
        }
    }

    pub fn provider_store_ready(
        provider_type: Option<String>,
        enabled_key_count: usize,
        total_key_count: usize,
    ) -> Self {
        Self {
            ready: true,
            status: "ready",
            source: "provider_store",
            reason_code: None,
            provider_type,
            enabled: Some(true),
            enabled_key_count: Some(enabled_key_count),
            total_key_count: Some(total_key_count),
            direct_request_config: false,
        }
    }

    pub fn to_payload(&self) -> Value {
        json!({
            "ready": self.ready,
            "status": self.status,
            "source": self.source,
            "reasonCode": self.reason_code,
            "reason_code": self.reason_code,
            "providerType": self.provider_type,
            "provider_type": self.provider_type,
            "enabled": self.enabled,
            "enabledKeyCount": self.enabled_key_count,
            "enabled_key_count": self.enabled_key_count,
            "totalKeyCount": self.total_key_count,
            "total_key_count": self.total_key_count,
            "directRequestConfig": self.direct_request_config,
            "direct_request_config": self.direct_request_config,
        })
    }
}

impl RoutingAttempt {
    pub(super) fn to_payload(&self) -> Value {
        json!({
            "slot": self.slot,
            "serviceModelSlot": self.slot,
            "service_model_slot": self.slot,
            "provider": self.provider,
            "model": self.model,
            "source": self.source,
            "providerReadiness": self.readiness.to_payload(),
            "provider_readiness": self.readiness.to_payload(),
            "runtimeFailure": self.runtime_failure.as_ref().map(ModelRouteExclusion::to_payload),
            "runtime_failure": self.runtime_failure.as_ref().map(ModelRouteExclusion::to_payload),
        })
    }
}

impl ModelRouteExclusion {
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        classification: FailureClassification,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            reason_code: runtime_failure_reason_code(classification),
            classification,
            retryable: true,
            credential_ref: None,
        }
    }

    pub fn for_credential(
        provider: impl Into<String>,
        model: impl Into<String>,
        credential_ref: impl Into<String>,
        classification: FailureClassification,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            reason_code: runtime_failure_reason_code(classification),
            classification,
            retryable: true,
            credential_ref: Some(credential_ref.into()),
        }
    }

    pub fn matches_route(&self, provider: &str, model: &str) -> bool {
        self.provider == provider && self.model == model
    }

    pub fn excludes_entire_route(&self) -> bool {
        self.credential_ref.is_none()
    }

    pub fn credential_ref(&self) -> Option<&str> {
        self.credential_ref.as_deref()
    }

    pub(super) fn to_payload(&self) -> Value {
        json!({
            "provider": self.provider,
            "model": self.model,
            "reasonCode": self.reason_code,
            "reason_code": self.reason_code,
            "classification": self.classification,
            "retryable": self.retryable,
        })
    }
}

impl fmt::Debug for ModelRouteExclusion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelRouteExclusion")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("reason_code", &self.reason_code)
            .field("classification", &self.classification)
            .field("retryable", &self.retryable)
            .field(
                "scope",
                &if self.credential_ref.is_some() {
                    "credential"
                } else {
                    "route"
                },
            )
            .finish()
    }
}

fn runtime_failure_reason_code(classification: FailureClassification) -> &'static str {
    match classification {
        FailureClassification::RateLimit => "provider_rate_limited",
        FailureClassification::ProviderInternal => "provider_internal_failure",
        FailureClassification::Transport => "provider_transport_failure",
        FailureClassification::Authentication
        | FailureClassification::Permission
        | FailureClassification::Quota
        | FailureClassification::InvalidRequest
        | FailureClassification::ContextOverflow
        | FailureClassification::ContentPolicy
        | FailureClassification::Unknown => "provider_runtime_failure",
    }
}
