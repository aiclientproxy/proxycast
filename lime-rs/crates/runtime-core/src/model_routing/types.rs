use serde_json::{json, Value};
use std::fmt;

use crate::FailureClassification;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CandidateRequirements {
    pub task_families: Vec<String>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub runtime_features: Vec<String>,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateModel {
    pub provider: String,
    pub model: String,
    pub source: String,
    pub status: String,
    pub task_families: Vec<String>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub runtime_features: Vec<String>,
    pub capabilities: Vec<String>,
    pub estimated_cost_class: Option<String>,
    pub limit_state: Option<String>,
    pub continuity_key: Option<String>,
}

impl CandidateModel {
    pub fn from_selection(selection: &RuntimeModelSelection) -> Self {
        Self {
            provider: selection.provider.clone(),
            model: selection.model.clone(),
            source: selection.source.to_string(),
            status: "unknown".to_string(),
            task_families: Vec::new(),
            input_modalities: Vec::new(),
            output_modalities: Vec::new(),
            runtime_features: Vec::new(),
            capabilities: Vec::new(),
            estimated_cost_class: None,
            limit_state: None,
            continuity_key: None,
        }
    }

    pub fn supports(&self, requirements: &CandidateRequirements) -> bool {
        contains_all(&self.task_families, &requirements.task_families)
            && contains_all(&self.input_modalities, &requirements.input_modalities)
            && contains_all(&self.output_modalities, &requirements.output_modalities)
            && contains_all(&self.runtime_features, &requirements.runtime_features)
            && requirements
                .capabilities
                .iter()
                .all(|required| capability_satisfied(required, self))
    }

    fn is_fallback_eligible(&self) -> bool {
        !matches!(self.status.as_str(), "deprecated" | "legacy")
            && !matches!(self.limit_state.as_deref(), Some("blocked" | "exhausted"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CandidateModelSet {
    pub requirements: CandidateRequirements,
    pub candidates: Vec<CandidateModel>,
}

impl CandidateModelSet {
    pub fn new(requirements: CandidateRequirements) -> Self {
        Self {
            requirements,
            candidates: Vec::new(),
        }
    }

    pub fn push_unique(&mut self, candidate: CandidateModel) {
        if let Some(existing) = self.candidates.iter_mut().find(|existing| {
            existing.provider == candidate.provider && existing.model == candidate.model
        }) {
            merge_candidate(existing, candidate);
            return;
        }
        self.candidates.push(candidate);
    }

    pub fn to_payload(&self) -> Value {
        json!({
            "requirements": {
                "taskFamilies": self.requirements.task_families,
                "task_families": self.requirements.task_families,
                "inputModalities": self.requirements.input_modalities,
                "input_modalities": self.requirements.input_modalities,
                "outputModalities": self.requirements.output_modalities,
                "output_modalities": self.requirements.output_modalities,
                "runtimeFeatures": self.requirements.runtime_features,
                "runtime_features": self.requirements.runtime_features,
                "capabilities": self.requirements.capabilities,
            },
            "candidates": self.candidates.iter().map(candidate_payload).collect::<Vec<_>>(),
        })
    }

    pub(super) fn resolve_candidates(
        &self,
        selection: &RuntimeModelSelection,
        policy: &OemRoutingPolicy,
        profile_slots: &[ProfileModelSlot],
    ) -> Vec<RuntimeModelSelection> {
        let mut candidates = Vec::new();
        if policy.allows_route(&selection.provider, &selection.model, false) {
            push_unique_selection(&mut candidates, selection.clone());
        }

        let selected_continuity = self
            .candidates
            .iter()
            .find(|candidate| {
                candidate.provider == selection.provider && candidate.model == selection.model
            })
            .and_then(|candidate| candidate.continuity_key.as_deref());

        let mut fallback_candidates = self
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.provider != selection.provider || candidate.model != selection.model
            })
            .filter(|candidate| candidate.is_fallback_eligible())
            .filter(|candidate| candidate.supports(&self.requirements))
            .filter(|candidate| policy.allows_route(&candidate.provider, &candidate.model, true))
            .collect::<Vec<_>>();
        fallback_candidates.sort_by_key(|candidate| {
            (
                selected_continuity
                    .is_some_and(|key| candidate.continuity_key.as_deref() == Some(key))
                    .then_some(0)
                    .unwrap_or(1),
                policy.preference_rank(&candidate.provider, &candidate.model),
                cost_rank(candidate.estimated_cost_class.as_deref()),
                profile_slot_rank(profile_slots, candidate),
            )
        });
        for candidate in fallback_candidates {
            push_unique_selection(
                &mut candidates,
                RuntimeModelSelection {
                    provider: candidate.provider.clone(),
                    model: candidate.model.clone(),
                    source: "candidate_model_set",
                    reasoning_effort: selection.reasoning_effort.clone(),
                },
            );
        }
        candidates
    }
}

fn contains_all(available: &[String], required: &[String]) -> bool {
    required.is_empty()
        || available.is_empty()
        || required.iter().all(|required| {
            available
                .iter()
                .any(|available| normalize_token(available) == normalize_token(required))
        })
}

fn capability_satisfied(required: &str, candidate: &CandidateModel) -> bool {
    if normalize_token(required) == "coding" {
        return true;
    }
    if candidate.capabilities.is_empty() {
        return true;
    }
    let required = normalize_token(required);
    let direct = candidate
        .capabilities
        .iter()
        .any(|capability| normalize_token(capability) == required);
    let tools_alias = required == "tools"
        && candidate
            .capabilities
            .iter()
            .any(|capability| normalize_token(capability) == "tool_calling");
    direct
        || tools_alias
        || (required == "vision"
            && candidate
                .input_modalities
                .iter()
                .any(|modality| normalize_token(modality) == "image"))
}

fn merge_candidate(existing: &mut CandidateModel, incoming: CandidateModel) {
    if existing.source == "runtime_options" || existing.source == "selection_derived" {
        existing.source = incoming.source.clone();
    }
    if existing.status == "unknown" {
        existing.status = incoming.status.clone();
    }
    merge_strings(&mut existing.task_families, incoming.task_families);
    merge_strings(&mut existing.input_modalities, incoming.input_modalities);
    merge_strings(&mut existing.output_modalities, incoming.output_modalities);
    merge_strings(&mut existing.runtime_features, incoming.runtime_features);
    merge_strings(&mut existing.capabilities, incoming.capabilities);
    if existing.estimated_cost_class.is_none() {
        existing.estimated_cost_class = incoming.estimated_cost_class;
    }
    if existing.limit_state.is_none() {
        existing.limit_state = incoming.limit_state;
    }
    if existing.continuity_key.is_none() {
        existing.continuity_key = incoming.continuity_key;
    }
}

fn merge_strings(existing: &mut Vec<String>, incoming: Vec<String>) {
    for value in incoming {
        if !existing.iter().any(|current| current == &value) {
            existing.push(value);
        }
    }
}

fn candidate_payload(candidate: &CandidateModel) -> Value {
    json!({
        "provider": candidate.provider,
        "model": candidate.model,
        "source": candidate.source,
        "status": candidate.status,
        "taskFamilies": candidate.task_families,
        "task_families": candidate.task_families,
        "inputModalities": candidate.input_modalities,
        "input_modalities": candidate.input_modalities,
        "outputModalities": candidate.output_modalities,
        "output_modalities": candidate.output_modalities,
        "runtimeFeatures": candidate.runtime_features,
        "runtime_features": candidate.runtime_features,
        "capabilities": candidate.capabilities,
        "estimatedCostClass": candidate.estimated_cost_class,
        "estimated_cost_class": candidate.estimated_cost_class,
        "limitState": candidate.limit_state,
        "limit_state": candidate.limit_state,
        "continuityKey": candidate.continuity_key,
        "continuity_key": candidate.continuity_key,
    })
}

fn profile_slot_rank(profile_slots: &[ProfileModelSlot], candidate: &CandidateModel) -> usize {
    profile_slots
        .iter()
        .position(|slot| {
            slot.provider.as_deref() == Some(candidate.provider.as_str())
                && slot.model.as_deref() == Some(candidate.model.as_str())
        })
        .unwrap_or(usize::MAX)
}

fn cost_rank(value: Option<&str>) -> u8 {
    match value {
        Some("free") => 0,
        Some("priced") => 1,
        _ => 2,
    }
}

fn normalize_token(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn push_unique_selection(
    candidates: &mut Vec<RuntimeModelSelection>,
    selection: RuntimeModelSelection,
) {
    if candidates.iter().any(|candidate| {
        candidate.provider == selection.provider && candidate.model == selection.model
    }) {
        return;
    }
    candidates.push(selection);
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OemRoutingMode {
    Managed,
    Hybrid,
    Advisory,
}

impl Default for OemRoutingMode {
    fn default() -> Self {
        Self::Advisory
    }
}

impl OemRoutingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Managed => "managed",
            Self::Hybrid => "hybrid",
            Self::Advisory => "advisory",
        }
    }
}

/// OEM constraints and preferences applied during candidate resolution.
///
/// The policy is intentionally provider/model agnostic. An allowlist entry may
/// be either a model id or a `provider/model` route identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OemRoutingPolicy {
    pub mode: OemRoutingMode,
    pub hard_model_allowlist: Vec<String>,
    pub soft_model_preferences: Vec<String>,
    pub fallback_to_local_allowed: bool,
}

impl Default for OemRoutingPolicy {
    fn default() -> Self {
        Self {
            mode: OemRoutingMode::Advisory,
            hard_model_allowlist: Vec::new(),
            soft_model_preferences: Vec::new(),
            fallback_to_local_allowed: true,
        }
    }
}

impl OemRoutingPolicy {
    pub fn is_configured(&self) -> bool {
        self.mode != OemRoutingMode::Advisory
            || !self.hard_model_allowlist.is_empty()
            || !self.soft_model_preferences.is_empty()
            || !self.fallback_to_local_allowed
    }

    pub fn allows_route(&self, provider: &str, model: &str, is_fallback: bool) -> bool {
        if !self.hard_model_allowlist.is_empty()
            && !self
                .hard_model_allowlist
                .iter()
                .any(|allowed| route_or_model_matches(allowed, provider, model))
        {
            return false;
        }
        if is_fallback && !self.fallback_to_local_allowed && is_local_route(provider, model) {
            return false;
        }
        true
    }

    pub fn preference_rank(&self, provider: &str, model: &str) -> usize {
        self.soft_model_preferences
            .iter()
            .position(|preferred| route_or_model_matches(preferred, provider, model))
            .unwrap_or(usize::MAX)
    }

    pub fn to_payload(&self) -> Value {
        json!({
            "mode": self.mode.as_str(),
            "routingMode": self.mode.as_str(),
            "routing_mode": self.mode.as_str(),
            "hardModelAllowlist": self.hard_model_allowlist,
            "hard_model_allowlist": self.hard_model_allowlist,
            "softModelPreferences": self.soft_model_preferences,
            "soft_model_preferences": self.soft_model_preferences,
            "fallbackToLocalAllowed": self.fallback_to_local_allowed,
            "fallback_to_local_allowed": self.fallback_to_local_allowed,
            "configured": self.is_configured(),
        })
    }
}

fn route_or_model_matches(value: &str, provider: &str, model: &str) -> bool {
    let value = normalize_policy_value(value);
    let provider = normalize_policy_value(provider);
    let model = normalize_policy_value(model);
    value == model
        || value == format!("{provider}/{model}")
        || value == format!("{provider}:{model}")
}

fn is_local_route(provider: &str, model: &str) -> bool {
    let provider = normalize_policy_value(provider);
    let model = normalize_policy_value(model);
    provider == "local"
        || provider == "ollama"
        || provider.contains("local")
        || model.starts_with("local/")
        || model.starts_with("local-")
}

fn normalize_policy_value(value: &str) -> String {
    value.trim().to_ascii_lowercase()
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
    pub candidate_count: u32,
    pub candidate_set: CandidateModelSet,
    pub oem_policy: OemRoutingPolicy,
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

    pub fn routing_policy_blocked(reason_code: &'static str) -> Self {
        Self {
            ready: false,
            status: "blocked",
            source: "routing_policy",
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
