//! Lime Skills Crate
//!
//! 包含 Skills 系统的 trait 定义和纯逻辑部分。
//! 宿主相关实现由 App Server 或 Desktop Host 边界承接。

#![allow(clippy::redundant_closure)]

mod agent_body;
mod agent_render;
mod agent_search;
mod agent_selection;
mod agent_snapshot;
mod execution_callback;
mod lime_llm_provider;
mod llm_provider;
mod orchestrator;
mod run;
mod skill_loader;
mod skill_matcher;
mod skill_summary;

// 电商 Skill 模块
pub mod ecommerce_review_reply;

pub use agent_body::{
    agent_skill_body_locator_from_metadata, estimate_agent_skill_body_tokens,
    evaluate_agent_skill_body, read_agent_skill_body, read_agent_skill_instructions,
    AgentSkillBody, AgentSkillBodyBudgetDecision, AgentSkillBodyBudgetDecisionKind,
    AgentSkillBodyEvaluation, AgentSkillBodyLocator, AgentSkillInstructions,
    DEFAULT_AGENT_SKILL_BODY_TOKEN_BUDGET,
};
pub use agent_render::{
    contains_agent_skills_prompt, contains_selected_agent_skill_body_prompt,
    render_available_agent_skills, render_selected_agent_skill_bodies, AgentSkillBodyRenderOptions,
    AgentSkillRenderOptions, DEFAULT_AGENT_SKILL_BODY_RENDER_CHAR_BUDGET,
    DEFAULT_AGENT_SKILL_RENDER_CHAR_BUDGET,
};
pub use agent_search::{
    reorder_agent_skill_snapshot_for_query, search_agent_skills, AgentSkillSearchOptions,
    AgentSkillSearchResult, DEFAULT_AGENT_SKILL_SEARCH_LIMIT,
};
pub use agent_selection::{
    evaluate_agent_skill_selection_bodies, select_agent_skills_by_name_candidates,
    select_explicit_agent_skills, select_implicit_agent_skills, select_mentioned_agent_skill,
    select_structured_agent_skill, AgentSkillSelection, AgentSkillSelectionEvaluation,
    AgentSkillSelectionTrigger,
};
pub use agent_snapshot::{
    agent_skill_roots_for_workspace, apply_agent_skill_config, build_agent_skill_snapshot,
    build_agent_skill_snapshot_from_roots, build_agent_skill_snapshot_from_workspace,
    default_agent_skill_roots, invalidate_agent_skill_snapshot_cache,
    set_runtime_extra_skill_roots, AgentSkillAuthority, AgentSkillDependencies,
    AgentSkillInterface, AgentSkillMetadata, AgentSkillPolicy, AgentSkillRoot, AgentSkillScope,
    AgentSkillSnapshot, AgentSkillSnapshotOptions, AgentSkillSource, AgentSkillToolDependency,
};
pub use execution_callback::{
    events, ExecutionCallback, ExecutionCompletePayload, StepCompletePayload, StepErrorPayload,
    StepStartPayload,
};
pub use lime_llm_provider::LimeLlmProvider;
pub use llm_provider::{LlmProvider, SkillError};
pub use orchestrator::{
    discover_orchestrator_skills, read_orchestrator_skill_resource,
    resource_belongs_to_orchestrator_package, OrchestratorSkillDiscovery,
    OrchestratorSkillResource, OrchestratorSkillResourceContent, OrchestratorSkillResourceGateway,
    OrchestratorSkillResourcePage, APPS_MCP_SERVER_NAME,
};
pub use run::{
    interpolate_variables, requires_turn_runtime, SkillRunResult, SkillRunner, SkillStepResult,
};
pub use skill_loader::{
    find_skill_by_name, get_lime_skills_dir, get_project_skills_dir, get_skill_roots,
    is_registered_skill, load_skill_from_file, load_skills_from_directory, parse_allowed_tools,
    parse_boolean, parse_skill_frontmatter, parse_workflow_steps, register_project_skill_directory,
    register_skill_directory, LoadedSkillDefinition, SkillFrontmatter, SkillTriggerConfig,
    WorkflowStep,
};
pub use skill_matcher::{SkillMatch, SkillMatcher};
pub use skill_summary::{
    load_skill_summaries_from_directory, load_skill_summary_from_file,
    load_skill_summary_report_from_directory, LoadedSkillSummary, SkillSummaryLoadError,
    SkillSummaryLoadReport,
};

pub const SKILL_SNAPSHOT_TURN_METADATA_KEY: &str = "skill_snapshot";
