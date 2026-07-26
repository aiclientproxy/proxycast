//! Lime Agent Crate
//!
//! 包含 Agent 模块中不依赖主 crate 内部模块的纯逻辑部分。
//! 深耦合部分（runtime_state、Agent 流式桥接）留在主 crate。

#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::to_string_in_format_args)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::borrowed_box)]

pub mod agent_tools;
pub mod artifact_protocol;
mod credential_bridge;
mod current_provider_turn;
mod direct_text_generation;
pub mod durable_memory_fs;
pub mod filesystem_event_protocol;
mod host_managed_generation;
mod knowledge_builder_skill;
pub mod lime_session_repository;
mod live_execution_process;
mod mcp_bridge;
mod model_request_policy;
pub mod native_tools;
pub mod prompt;
pub mod protocol;
mod protocol_context_projection;
pub mod protocol_projection;
mod provider_configuration;
pub mod provider_continuation_state;
pub mod provider_runtime_governor;
mod request_tool_policy;
mod request_user_input_bridge;
pub mod runtime_projection_snapshot;
mod runtime_state;
mod runtime_state_support;
mod session_configuration;
mod session_execution_runtime;
mod session_record_sql;
mod session_usage_projection;
pub mod skill_execution;
pub mod tool_io_offload;
pub mod tools;
mod turn_context_configuration;
mod turn_execution;
pub mod turn_input_envelope;
pub mod turn_state;
mod write_artifact_events;

pub use direct_text_generation::{
    run_direct_text_generation_with_db, DirectTextGenerationRequest, DirectTextGenerationResult,
};
pub use durable_memory_fs::{
    durable_memory_permission_pattern, is_virtual_memory_path, resolve_durable_memory_root,
    resolve_virtual_memory_path, to_virtual_memory_path, virtual_memory_relative_path,
    DURABLE_MEMORY_VIRTUAL_ROOT, LEGACY_DURABLE_MEMORY_ROOT_ENV, LIME_DURABLE_MEMORY_ROOT_ENV,
};
pub use host_managed_generation::{
    host_managed_generation_session_id, run_host_managed_generation,
    write_host_managed_generation_status, HostManagedGenerationPlan,
    HostManagedGenerationRunRequest, HostManagedGenerationRunResult,
    HOST_MANAGED_GENERATION_SCHEMA, HOST_MANAGED_GENERATION_SOURCE,
};
pub use knowledge_builder_skill::{
    run_knowledge_builder_skill, KnowledgeBuilderSkillRequest, KnowledgeBuilderSkillRunner,
};
pub use lime_mcp as mcp;
pub use live_execution_process::LiveExecutionProcessGateway;
pub use model_request_policy::{
    model_request_policy_from_metadata, model_request_policy_from_turn_context,
    native_tool_policy_disallowed_tool_names, native_tool_policy_from_metadata,
    native_tool_policy_from_turn_context, runtime_reply_model_request_policy_from_metadata,
    runtime_reply_model_request_policy_from_snapshot,
    runtime_reply_model_request_policy_from_turn_context, ModelNativeToolPolicySnapshot,
    ModelRequestPolicySnapshot, ModelResponsesPolicySnapshot, ModelToolCallPolicySnapshot,
    ModelTruncationPolicySnapshot, MODEL_NATIVE_APPLY_PATCH_TOOL_NAME,
    MODEL_NATIVE_SHELL_TOOL_NAME, MODEL_NATIVE_STDIN_TOOL_NAME,
};
pub use prompt::{
    budget_limit_prompt, build_runtime_agents_prompt, build_runtime_agents_prompt_for_project,
    continuation_prompt, merge_system_prompt_with_runtime_agents,
    merge_system_prompt_with_runtime_agents_for_project, objective_updated_prompt,
    permissions_instructions, render_review_exit_interrupted, render_review_exit_success,
    resolve_review_prompt, review_prompt, PermissionsPromptInput, PromptApprovalPolicy,
    PromptNetworkAccess, PromptSandboxMode, ResolvedReviewPrompt, ReviewPromptTarget,
    ThreadGoalPromptInput, HIERARCHICAL_AGENTS_MESSAGE, REALTIME_BACKEND_PROMPT,
    REALTIME_END_INSTRUCTIONS, REALTIME_START_INSTRUCTIONS, REVIEW_PROMPT,
    RUNTIME_AGENTS_PROMPT_MARKER, SUMMARIZATION_PROMPT, SUMMARY_PREFIX,
};
pub use protocol::{
    build_diagnostics_runtime_status_metadata, AgentActionRequiredScope, AgentArtifactSignal,
    AgentEvent, AgentImageAttachment, AgentMessage, AgentMessageContent, AgentOp,
    AgentProviderTraceEvent, AgentProviderTraceStage, AgentRuntimeStatus, AgentTokenUsage,
    AgentToolImage, AgentToolProgressPayload, AgentToolResult, AgentUserInputOp,
    AgentUserPreferences, TextDeltaBatchBoundary,
};
pub use protocol_projection::{project_item_runtime, project_turn_runtime};
pub use provider_configuration::{
    route_protocol_from_session_provider_config, ModelRouteProviderConfiguration,
    SessionProviderConfig,
};
pub use provider_continuation_state::{
    ProviderContinuationCapability, ProviderContinuationCapable, ProviderContinuationState,
};
pub use provider_runtime_governor::{
    acquire_provider_runtime_permit, preview_provider_runtime_wait_snapshot,
    release_provider_runtime_permit, resolve_provider_runtime_parallel_budget,
    snapshot_provider_runtime_lease, ProviderRuntimeGovernorSnapshot, ProviderRuntimePermit,
};
pub use request_tool_policy::{
    merge_system_prompt_with_request_tool_policy,
    request_tool_policy_with_additional_required_tools, resolve_request_tool_policy,
    resolve_request_tool_policy_with_mode, stream_reply_with_policy, ReplyAttemptError,
    RequestToolPolicy, RequestToolPolicyMode, StreamReplyExecution, WebSearchExecutionTracker,
    REQUEST_TOOL_POLICY_MARKER,
};
pub use runtime_projection_snapshot::RuntimeProjectionSnapshot;
pub use runtime_state::AgentRuntimeState;
pub use runtime_state_support::{
    is_skill_registered, register_project_skill_from_directory, reload_skills,
};
pub use session_configuration::{
    build_agent_session_config, AgentSessionConfig, AgentSessionConfigurationRequest,
    SessionConfigBuilder,
};
pub use session_execution_runtime::{
    SessionExecutionRuntimeCostState, SessionExecutionRuntimeLimitEvent,
    SessionExecutionRuntimeLimitState, SessionExecutionRuntimeRoutingDecision,
    SessionExecutionRuntimeTaskProfile,
};
pub use skill_execution::{
    execute_skill_prompt, execute_skill_workflow, SkillEventEmitter, SkillExecutionError,
    SkillExecutionResult, SkillInputImage, SkillPromptExecution, SkillWorkflowExecution,
    StepResult,
};
pub use turn_context_configuration::{
    agent_turn_approval_policy, agent_turn_context_metadata, agent_turn_sandbox_policy,
    build_agent_turn_context, insert_agent_turn_metadata, set_agent_turn_output_schema,
    set_agent_turn_user_visible_input_text, AgentTurnContext, AgentTurnContextConfigurationRequest,
    AgentTurnContextOverride,
};
pub use turn_execution::{
    run_agent_turn_with_policy, AgentTurnExecution, AgentTurnExecutionRequest,
    AgentTurnProviderConfiguration,
};
pub use turn_input_envelope::{
    TurnDiagnosticsSnapshot, TurnExecutionProfile, TurnInputEnvelope, TurnInputEnvelopeBuilder,
    TurnMessageHistorySource, TurnPromptAugmentationStage, TurnPromptAugmentationStageKind,
    TurnProviderRoutingSnapshot, TurnRequestToolPolicySnapshot, TurnSystemPromptSource,
};
pub use turn_state::TurnState;
pub use write_artifact_events::WriteArtifactEventEmitter;
