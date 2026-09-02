//! Skill 执行回调 trait 和 Payload 类型
//!
//! 定义 Skill 执行过程中的回调接口和事件数据类型。
//! 具体宿主实现由 App Server 或 Desktop Host 边界注入。

use serde::Serialize;

/// 步骤开始事件 Payload
#[derive(Debug, Clone, Serialize)]
pub struct StepStartPayload {
    pub execution_id: String,
    pub step_id: String,
    pub step_name: String,
    pub current_step: usize,
    pub total_steps: usize,
}

/// 步骤完成事件 Payload
#[derive(Debug, Clone, Serialize)]
pub struct StepCompletePayload {
    pub execution_id: String,
    pub step_id: String,
    pub output: String,
}

/// 步骤错误事件 Payload
#[derive(Debug, Clone, Serialize)]
pub struct StepErrorPayload {
    pub execution_id: String,
    pub step_id: String,
    pub error: String,
    pub will_retry: bool,
}

/// 执行完成事件 Payload
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionCompletePayload {
    pub execution_id: String,
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
}

/// Skill 执行事件名称常量
pub mod events {
    pub const STEP_START: &str = "skill:step_start";
    pub const STEP_COMPLETE: &str = "skill:step_complete";
    pub const STEP_ERROR: &str = "skill:step_error";
    pub const COMPLETE: &str = "skill:complete";
}

/// ExecutionCallback Trait
///
/// 定义 Skill 执行过程中的回调接口。
/// 应用层需要实现此 trait 以接收执行进度更新。
pub trait ExecutionCallback: Send + Sync {
    fn on_step_start(
        &self,
        step_id: &str,
        step_name: &str,
        current_step: usize,
        total_steps: usize,
    );

    fn on_step_complete(&self, step_id: &str, output: &str);

    fn on_step_error(&self, step_id: &str, error: &str, will_retry: bool);

    fn on_complete(&self, success: bool, final_output: Option<&str>, error: Option<&str>);
}
