//! AI Agent 集成模块
//!
//! 提供基于 OpenAI 兼容 API 的 Agent 实现
//! 包含工具系统、流式处理和工具调用循环

pub mod native_agent;
pub mod tool_loop;
pub mod tools;
pub mod types;

pub use native_agent::{NativeAgent, NativeAgentState};
pub use tool_loop::{ToolCallResult, ToolLoopConfig, ToolLoopEngine, ToolLoopError, ToolLoopState};
pub use types::*;
