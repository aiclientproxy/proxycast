pub mod context_fragments;
pub mod llm_protocol;
pub mod model_route;
pub mod model_routing;
pub mod model_task;
pub mod runtime_content;

pub use context_fragments::{
    estimate_context_fragment_tokens, ContextFragmentBudgetDecision, ContextFragmentBudgetPolicy,
    ContextFragmentBudgetStatus, ContextFragmentEnvelope, ContextFragmentInput,
    ContextFragmentSource, ContextSidecarReference,
};
pub use llm_protocol::canonical::{
    ContentPart, FailureClassification, FinishReason, FreeformToolFormat, GenerationOptions,
    LlmEvent as CanonicalLlmEvent, Message as CanonicalMessage, ModelRerouteReason,
    ModelVerification, ProviderMetadata, Request as CanonicalRequest, Role as CanonicalRole,
    ToolDefinition as CanonicalToolDefinition, ToolResultValue, Usage,
};
pub use model_route::{
    protocol_from_provider_name, resolved_route_from_task, route_evidence_payload,
    route_resolution_evidence_payloads, DirectRouteConfig, ModelRouteProvider, ModelRouteSelection,
    RouteResolutionEvidencePayloads,
};
pub use model_routing::{
    resolve_model_routing_for_candidate, resolve_ready_model_routing,
    resolve_ready_model_routing_with_candidate_set, resolve_ready_model_routing_with_exclusions,
    routing_decision_payload, routing_fallback_applied_payload, routing_not_possible_payload,
    routing_not_possible_payload_with_attempts, selection_from_profile_model_slot, CandidateModel,
    CandidateModelSet, CandidateRequirements, ModelRouteExclusion, ModelRoutingDecision,
    OemRoutingMode, OemRoutingPolicy, ProfileModelSlot, ProviderReadiness, RoutingAttempt,
    RoutingResolution, RuntimeModelSelection, PROFILE_MODEL_SLOT_SOURCE,
};
pub use model_task::{
    build_model_task_request, capability_snapshot_from_model_capabilities,
    model_task_request_value, route_capability_gap, ModelTaskRequestInput,
};
pub use runtime_content::{
    is_supported_runtime_media_mime_type, runtime_media_kind_for_mime_type,
    runtime_media_part_from_reference, RuntimeContentPart, RuntimeContentPartError,
    RuntimeContentReference, RuntimeMediaKind, RuntimeMediaPartInput, RuntimeMessageDeltaContent,
    RuntimeMessageDeltaContentParseError, RUNTIME_AUDIO_MIME_TYPES, RUNTIME_FILE_MIME_TYPES,
    RUNTIME_IMAGE_MIME_TYPES, RUNTIME_VIDEO_MIME_TYPES,
};
