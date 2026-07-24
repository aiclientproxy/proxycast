//! Current runtime prompt assets and Codex-compatible AGENTS instruction loading.

pub mod prompt_assets;
pub mod runtime_agents;

pub use prompt_assets::*;
pub use runtime_agents::{
    build_runtime_agents_prompt, build_runtime_agents_prompt_for_project,
    merge_system_prompt_with_runtime_agents, merge_system_prompt_with_runtime_agents_for_project,
    RUNTIME_AGENTS_PROMPT_MARKER,
};
