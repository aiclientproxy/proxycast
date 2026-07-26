mod payload;
mod profile_slots;
#[cfg(test)]
mod tests;
mod types;

pub use payload::{
    routing_decision_payload, routing_fallback_applied_payload, routing_not_possible_payload,
    routing_not_possible_payload_with_attempts,
};
pub use profile_slots::{
    resolve_model_routing_for_candidate, resolve_ready_model_routing,
    resolve_ready_model_routing_with_exclusions, selection_from_profile_model_slot,
};
pub use types::{
    ModelRouteExclusion, ModelRoutingDecision, ProfileModelSlot, ProviderReadiness, RoutingAttempt,
    RoutingResolution, RuntimeModelSelection,
};

pub const PROFILE_MODEL_SLOT_SOURCE: &str = "profile_model_slot";

pub(super) const DEFAULT_CODING_SLOT: &str = "coding";
pub(super) const DERIVED_MODEL_SLOT_SOURCE: &str = "selection_derived";
pub(super) const REQUIRED_CODING_CAPABILITIES: &[&str] = &["coding", "tools", "streaming"];
pub(super) const KNOWN_CODING_SLOTS: &[&str] = &["base", "coding", "review", "fast", "local"];
