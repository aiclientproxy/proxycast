//! 管道步骤 trait 定义

#![allow(dead_code)]

use async_trait::async_trait;
use lime_core::processor::RequestContext;
use thiserror::Error;

/// 步骤错误
#[derive(Error, Debug, Clone)]
pub enum StepError {
    #[error("认证错误: {0}")]
    Auth(String),
    #[error("路由错误: {0}")]
    Routing(String),
    #[error("注入错误: {0}")]
    Injection(String),
    #[error("Provider 错误: {0}")]
    Provider(String),
    #[error("遥测错误: {0}")]
    Telemetry(String),
    #[error("超时: {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    #[error("内部错误: {0}")]
    Internal(String),
}

impl StepError {
    /// 获取对应的 HTTP 状态码
    pub fn status_code(&self) -> u16 {
        match self {
            StepError::Auth(_) => 401,
            StepError::Routing(_) => 404,
            StepError::Injection(_) => 400,
            StepError::Provider(_) => 502,
            StepError::Telemetry(_) => 500,
            StepError::Timeout { .. } => 408,
            StepError::Internal(_) => 500,
        }
    }
}

/// 管道步骤 trait
#[async_trait]
pub trait PipelineStep: Send + Sync {
    async fn execute(
        &self,
        ctx: &mut RequestContext,
        payload: &mut serde_json::Value,
    ) -> Result<(), StepError>;

    fn name(&self) -> &str;

    fn is_enabled(&self) -> bool {
        true
    }
}
