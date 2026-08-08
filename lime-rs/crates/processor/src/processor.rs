//! 请求处理器实现
//!
//! 提供统一的请求处理管道，集成路由、容错、监控、插件等功能模块。
//!
//! # 架构
//!
//! 请求处理流程：
//! 1. 认证 (AuthStep)
//! 2. 参数注入 (InjectionStep)
//! 3. 路由解析 (RoutingStep)
//! 4. Provider 调用 (ProviderStep) - 包含重试和故障转移
//! 5. 统计记录 (TelemetryStep)

pub use lime_core::processor::RequestContext;

use lime_core::router::{ModelMapper, Router};
use lime_core::ProviderType;
use lime_infra::{Failover, Injector, Retrier, StatsAggregator, TimeoutController, TokenTracker};
use parking_lot::RwLock as ParkingLotRwLock;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 统一的请求处理器
///
/// 集成所有功能模块，提供完整的请求处理管道
pub struct RequestProcessor {
    /// 路由器
    pub router: Arc<RwLock<Router>>,
    /// 模型映射器
    pub mapper: Arc<RwLock<ModelMapper>>,
    /// 参数注入器
    pub injector: Arc<RwLock<Injector>>,
    /// 重试器
    pub retrier: Arc<Retrier>,
    /// 故障转移器
    pub failover: Arc<Failover>,
    /// 超时控制器
    pub timeout: Arc<TimeoutController>,
    /// 统计聚合器（使用 parking_lot::RwLock 以支持与 TelemetryState 共享）
    pub stats: Arc<ParkingLotRwLock<StatsAggregator>>,
    /// Token 追踪器（使用 parking_lot::RwLock 以支持与 TelemetryState 共享）
    pub tokens: Arc<ParkingLotRwLock<TokenTracker>>,
    /// 热重载协调锁（避免配置更新期间请求读取不一致的配置）
    pub reload_lock: Arc<RwLock<()>>,
    /// 提示路由器
    pub hint_router: Arc<RwLock<lime_core::router::HintRouter>>,
    /// 对话修剪器
    pub conversation_trimmer: Arc<crate::conversation_manager::ConversationTrimmer>,
}

impl RequestProcessor {
    /// 创建新的请求处理器
    pub fn new(
        router: Arc<RwLock<Router>>,
        mapper: Arc<RwLock<ModelMapper>>,
        injector: Arc<RwLock<Injector>>,
        retrier: Arc<Retrier>,
        failover: Arc<Failover>,
        timeout: Arc<TimeoutController>,
        stats: Arc<ParkingLotRwLock<StatsAggregator>>,
        tokens: Arc<ParkingLotRwLock<TokenTracker>>,
    ) -> Self {
        Self {
            router,
            mapper,
            injector,
            retrier,
            failover,
            timeout,
            stats,
            tokens,
            reload_lock: Arc::new(RwLock::new(())),
            hint_router: Arc::new(RwLock::new(lime_core::router::HintRouter::default())),
            conversation_trimmer: Arc::new(crate::conversation_manager::ConversationTrimmer::new(
                crate::conversation_manager::TrimConfig::default(),
            )),
        }
    }

    /// 使用默认配置创建请求处理器
    pub fn with_defaults() -> Self {
        Self {
            router: Arc::new(RwLock::new(Self::create_router_with_defaults())),
            mapper: Arc::new(RwLock::new(ModelMapper::new())),
            injector: Arc::new(RwLock::new(Injector::new())),
            retrier: Arc::new(Retrier::with_defaults()),
            failover: Arc::new(Failover::with_defaults()),
            timeout: Arc::new(TimeoutController::with_defaults()),
            stats: Arc::new(ParkingLotRwLock::new(StatsAggregator::with_defaults())),
            tokens: Arc::new(ParkingLotRwLock::new(TokenTracker::with_defaults())),
            reload_lock: Arc::new(RwLock::new(())),
            hint_router: Arc::new(RwLock::new(lime_core::router::HintRouter::default())),
            conversation_trimmer: Arc::new(crate::conversation_manager::ConversationTrimmer::new(
                crate::conversation_manager::TrimConfig::default(),
            )),
        }
    }

    /// 创建带默认路由规则的路由器
    ///
    /// 注意：不再添加硬编码的路由规则，让用户设置的默认 Provider 生效
    fn create_router_with_defaults() -> Router {
        let router = Router::new_empty();
        tracing::info!("[ROUTER] 初始化空路由器，等待从配置加载默认 Provider");
        router
    }

    /// 使用共享的统计和 Token 追踪器创建请求处理器
    pub fn with_shared_telemetry(
        stats: Arc<ParkingLotRwLock<StatsAggregator>>,
        tokens: Arc<ParkingLotRwLock<TokenTracker>>,
    ) -> Self {
        Self {
            router: Arc::new(RwLock::new(Self::create_router_with_defaults())),
            mapper: Arc::new(RwLock::new(ModelMapper::new())),
            injector: Arc::new(RwLock::new(Injector::new())),
            retrier: Arc::new(Retrier::with_defaults()),
            failover: Arc::new(Failover::with_defaults()),
            timeout: Arc::new(TimeoutController::with_defaults()),
            stats,
            tokens,
            reload_lock: Arc::new(RwLock::new(())),
            hint_router: Arc::new(RwLock::new(lime_core::router::HintRouter::default())),
            conversation_trimmer: Arc::new(crate::conversation_manager::ConversationTrimmer::new(
                crate::conversation_manager::TrimConfig::default(),
            )),
        }
    }

    /// 解析模型别名
    pub async fn resolve_model(&self, model: &str) -> String {
        let mapper = self.mapper.read().await;
        mapper.resolve(model)
    }

    /// 解析模型别名并更新请求上下文
    pub async fn resolve_model_for_context(&self, ctx: &mut RequestContext) -> String {
        let resolved = self.resolve_model(&ctx.original_model).await;
        ctx.set_resolved_model(resolved.clone());

        tracing::debug!(
            "[MAPPER] request_id={} original_model={} resolved_model={}",
            ctx.request_id,
            ctx.original_model,
            resolved
        );

        resolved
    }

    /// 根据模型选择 Provider
    pub async fn route_model(&self, model: &str) -> (Option<ProviderType>, bool) {
        let router = self.router.read().await;
        let result = router.route(model);
        (result.provider, result.is_default)
    }

    /// 根据模型选择 Provider 并更新请求上下文
    pub async fn route_for_context(&self, ctx: &mut RequestContext) -> Option<ProviderType> {
        let (provider, is_default) = self.route_model(&ctx.resolved_model).await;

        if let Some(p) = provider {
            ctx.set_provider(p);
            tracing::info!(
                "[ROUTE] request_id={} model={} provider={} is_default={}",
                ctx.request_id,
                ctx.resolved_model,
                p,
                is_default
            );
        } else {
            tracing::warn!(
                "[ROUTE] request_id={} model={} 未设置默认 Provider",
                ctx.request_id,
                ctx.resolved_model
            );
        }

        provider
    }

    /// 执行完整的路由解析流程（模型别名解析 + Provider 选择）
    pub async fn resolve_and_route(&self, ctx: &mut RequestContext) -> Option<ProviderType> {
        self.resolve_model_for_context(ctx).await;
        self.route_for_context(ctx).await
    }
}
