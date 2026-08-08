//! 请求处理器 crate
//!
//! 提供统一的请求处理管道，集成路由、容错、监控、插件等功能模块。
//!
//! ## 模块结构

#![allow(clippy::derivable_impls)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::too_many_arguments)]
//!
//! - `steps` - 管道步骤（认证、注入、路由、Provider、遥测）

pub mod conversation_manager;
pub mod conversation_summarizer;
pub mod processor;
pub mod steps;

pub use lime_core::processor::RequestContext;
pub use processor::RequestProcessor;
pub use steps::*;
