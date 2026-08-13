//! Lime Agent Scheduler
//!
//! 提供 Agent 任务调度功能，支持定时任务、重试机制等。
//!
//! ## 功能

#![allow(clippy::redundant_closure)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::useless_format)]
#![allow(clippy::derivable_impls)]
//! - 任务创建和管理
//! - 任务持久化到 SQLite
//! - 定时任务调度
//! - 任务状态跟踪
//! - 失败重试机制
//!
//! ## 使用示例
//!
//! ```rust,no_run
//! use lime_scheduler::{AgentScheduler, ScheduledTask, SchedulerTrait};
//! use chrono::Utc;
//!
//! # async fn example(db: lime_core::database::DbConnection) -> Result<(), String> {
//! // 初始化调度器
//! AgentScheduler::init_tables(&db)?;
//! let scheduler = AgentScheduler::new(db);
//!
//! // 创建任务
//! let task = ScheduledTask::new(
//!     "测试任务".to_string(),
//!     "test_task".to_string(),
//!     serde_json::json!({"param": "value"}),
//!     "openai".to_string(),
//!     "gpt-4".to_string(),
//!     Utc::now(),
//! );
//!
//! let task_id = scheduler.create_task(task).await?;
//!
//! // 获取到期任务
//! let due_tasks = scheduler.get_due_tasks(10).await?;
//! # Ok(())
//! # }
//! ```

pub mod claim;
pub mod dao;
pub mod scheduler;
#[cfg(test)]
mod task_context;
pub mod types;

pub use claim::{
    AutomationOwnedRun, AutomationRunRecovery, AutomationRunRecoveryKind,
    AutomationRunScheduleRecovery, AutomationRunTerminal, AutomationWindowClaim,
    AutomationWindowClaimDao,
};
pub use dao::SchedulerDao;
pub use scheduler::{AgentScheduler, SchedulerGovernanceConfig, SchedulerTrait};
pub use types::{
    ScheduledTask, TaskFilter, TaskStatus, DEFAULT_TASK_COOLDOWN_SECS,
    DEFAULT_TASK_FAILURE_THRESHOLD,
};
