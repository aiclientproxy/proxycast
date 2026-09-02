//! 基础设施模块
//!
//! 包含独立的基础设施组件，不依赖业务逻辑：
//! - proxy: HTTP 代理客户端
//! - resilience: 重试、熔断、故障转移
//! - injection: 请求参数注入
//! - telemetry: 遥测统计
//!
pub mod injection;
pub mod process;
pub mod proxy;
pub mod resilience;
pub mod telemetry;

// 重新导出常用类型
pub use injection::{InjectionConfig, InjectionMode, InjectionResult, InjectionRule, Injector};
pub use process::{configure_std_command_for_gui, configure_tokio_command_for_gui};
pub use proxy::{ProxyClientFactory, ProxyError, ProxyProtocol};
pub use resilience::{
    Failover, FailoverConfig, Retrier, RetryConfig, TimeoutConfig, TimeoutController,
};
pub use telemetry::{
    LogRotationConfig, LoggerError, ModelStats, ModelTokenStats, PeriodTokenStats, ProviderStats,
    ProviderTokenStats, RequestLog, RequestLogger, RequestStatus, StatsAggregator, StatsSummary,
    TelemetryStore, TimeRange, TokenSource, TokenStatsSummary, TokenTracker, TokenUsageRecord,
};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
