//! 请求处理器核心类型
//!
//! 包含请求上下文和处理错误类型定义。
//! 完整的请求处理管道（步骤、路由、Provider 集成等）保留在主 crate 中。

pub mod context;
pub mod error;

pub use context::RequestContext;
pub use error::ProcessError;
