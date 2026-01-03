//! HTTP API 服务器

pub mod client_detector;

use crate::config::{
    Config, ConfigChangeEvent, ConfigChangeKind, ConfigManager, EndpointProvidersConfig,
    FileWatcher, HotReloadManager, ReloadResult,
};
use crate::converter::anthropic_to_openai::convert_anthropic_to_openai;
use crate::credential::CredentialSyncService;
use crate::database::dao::provider_pool::ProviderPoolDao;
use crate::database::DbConnection;
use crate::flow_monitor::{FlowInterceptor, FlowMonitor, FlowMonitorConfig};
use crate::injection::Injector;
use crate::logger::LogStore;
use crate::models::anthropic::*;
use crate::models::openai::*;
use crate::models::provider_pool_model::CredentialData;
use crate::models::route_model::{RouteInfo, RouteListResponse};
use crate::processor::{RequestContext, RequestProcessor};
use crate::providers::antigravity::AntigravityProvider;
use crate::providers::claude_custom::ClaudeCustomProvider;
use crate::providers::gemini::GeminiProvider;
use crate::providers::kiro::KiroProvider;
use crate::providers::openai_custom::OpenAICustomProvider;
use crate::providers::qwen::QwenProvider;
use crate::server_utils::{
    build_anthropic_response, build_anthropic_stream_response, build_gemini_native_request, health,
    models, parse_cw_response,
};
use crate::services::kiro_event_service::KiroEventService;
use crate::services::provider_pool_service::ProviderPoolService;
use crate::services::token_cache_service::TokenCacheService;
use crate::websocket::{WsConfig, WsConnectionManager, WsStats};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{oneshot, RwLock};

/// 记录请求统计到遥测系统
pub fn record_request_telemetry(
    state: &AppState,
    ctx: &RequestContext,
    status: crate::telemetry::RequestStatus,
    error_message: Option<String>,
) {
    use crate::telemetry::RequestLog;

    let provider = ctx.provider.unwrap_or(crate::ProviderType::Kiro);
    let mut log = RequestLog::new(
        ctx.request_id.clone(),
        provider,
        ctx.resolved_model.clone(),
        ctx.is_stream,
    );

    // 设置状态和持续时间
    match status {
        crate::telemetry::RequestStatus::Success => log.mark_success(ctx.elapsed_ms(), 200),
        crate::telemetry::RequestStatus::Failed => log.mark_failed(
            ctx.elapsed_ms(),
            None,
            error_message.clone().unwrap_or_default(),
        ),
        crate::telemetry::RequestStatus::Timeout => log.mark_timeout(ctx.elapsed_ms()),
        crate::telemetry::RequestStatus::Cancelled => log.mark_cancelled(ctx.elapsed_ms()),
        crate::telemetry::RequestStatus::Retrying => {
            log.duration_ms = ctx.elapsed_ms();
        }
    }

    // 设置凭证 ID
    if let Some(cred_id) = &ctx.credential_id {
        log.set_credential_id(cred_id.clone());
    }

    // 设置重试次数
    log.retry_count = ctx.retry_count;

    // 记录到统计聚合器
    {
        let stats = state.processor.stats.write();
        stats.record(log.clone());
    }

    // 记录到请求日志记录器（用于前端日志列表显示）
    if let Some(logger) = &state.request_logger {
        let _ = logger.record(log.clone());
    }

    tracing::info!(
        "[TELEMETRY] request_id={} provider={:?} model={} status={:?} duration_ms={}",
        ctx.request_id,
        provider,
        ctx.resolved_model,
        status,
        ctx.elapsed_ms()
    );
}

/// 记录 Token 使用量到遥测系统
pub fn record_token_usage(
    state: &AppState,
    ctx: &RequestContext,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
) {
    use crate::telemetry::{TokenSource, TokenUsageRecord};

    // 只有当至少有一个 Token 值时才记录
    if input_tokens.is_none() && output_tokens.is_none() {
        return;
    }

    let provider = ctx.provider.unwrap_or(crate::ProviderType::Kiro);
    let record = TokenUsageRecord::new(
        uuid::Uuid::new_v4().to_string(),
        provider,
        ctx.resolved_model.clone(),
        input_tokens.unwrap_or(0),
        output_tokens.unwrap_or(0),
        TokenSource::Actual,
    )
    .with_request_id(ctx.request_id.clone());

    // 记录到 Token 追踪器
    {
        let tokens = state.processor.tokens.write();
        tokens.record(record);
    }

    tracing::debug!(
        "[TOKEN] request_id={} input={} output={}",
        ctx.request_id,
        input_tokens.unwrap_or(0),
        output_tokens.unwrap_or(0)
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    pub running: bool,
    pub host: String,
    pub port: u16,
    pub requests: u64,
    pub uptime_secs: u64,
}

pub struct ServerState {
    pub config: Config,
    pub running: bool,
    pub requests: u64,
    pub start_time: Option<std::time::Instant>,
    pub kiro_provider: KiroProvider,
    pub gemini_provider: GeminiProvider,
    pub qwen_provider: QwenProvider,
    pub openai_custom_provider: OpenAICustomProvider,
    pub claude_custom_provider: ClaudeCustomProvider,
    pub default_provider_ref: Arc<RwLock<String>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// 服务器运行时使用的 API key（启动时从配置复制）
    /// 用于 test_api 命令，确保测试使用的 API key 和服务器一致
    pub running_api_key: Option<String>,
}

impl ServerState {
    pub fn new(config: Config) -> Self {
        let kiro = KiroProvider::new();
        let gemini = GeminiProvider::new();
        let qwen = QwenProvider::new();
        let openai_custom = OpenAICustomProvider::new();
        let claude_custom = ClaudeCustomProvider::new();
        let default_provider_ref = Arc::new(RwLock::new(config.default_provider.clone()));

        Self {
            config,
            running: false,
            requests: 0,
            start_time: None,
            kiro_provider: kiro,
            gemini_provider: gemini,
            qwen_provider: qwen,
            openai_custom_provider: openai_custom,
            claude_custom_provider: claude_custom,
            default_provider_ref,
            shutdown_tx: None,
            running_api_key: None,
        }
    }

    pub fn status(&self) -> ServerStatus {
        ServerStatus {
            running: self.running,
            host: self.config.server.host.clone(),
            port: self.config.server.port,
            requests: self.requests,
            uptime_secs: self.start_time.map(|t| t.elapsed().as_secs()).unwrap_or(0),
        }
    }

    /// 增加请求计数
    pub fn increment_request_count(&mut self) {
        self.requests = self.requests.saturating_add(1);
    }

    pub async fn start(
        &mut self,
        logs: Arc<RwLock<LogStore>>,
        pool_service: Arc<ProviderPoolService>,
        token_cache: Arc<TokenCacheService>,
        db: Option<DbConnection>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.start_with_telemetry(logs, pool_service, token_cache, db, None, None, None)
            .await
    }

    /// 启动服务器（使用共享的遥测实例）
    ///
    /// 这允许服务器与 TelemetryState 共享同一个 StatsAggregator、TokenTracker 和 RequestLogger，
    /// 使得请求处理过程中记录的统计数据能够在前端监控页面中显示。
    pub async fn start_with_telemetry(
        &mut self,
        logs: Arc<RwLock<LogStore>>,
        pool_service: Arc<ProviderPoolService>,
        token_cache: Arc<TokenCacheService>,
        db: Option<DbConnection>,
        shared_stats: Option<Arc<parking_lot::RwLock<crate::telemetry::StatsAggregator>>>,
        shared_tokens: Option<Arc<parking_lot::RwLock<crate::telemetry::TokenTracker>>>,
        shared_logger: Option<Arc<crate::telemetry::RequestLogger>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.start_with_telemetry_and_flow_monitor(
            logs,
            pool_service,
            token_cache,
            db,
            shared_stats,
            shared_tokens,
            shared_logger,
            None,
            None,
        )
        .await
    }

    /// 启动服务器（使用共享的遥测实例和 Flow Monitor）
    ///
    /// 这允许服务器与 TelemetryState 共享同一个 StatsAggregator、TokenTracker 和 RequestLogger，
    /// 以及与 FlowMonitorState 共享同一个 FlowMonitor，
    /// 使得请求处理过程中记录的统计数据和 Flow 数据能够在前端监控页面中显示。
    pub async fn start_with_telemetry_and_flow_monitor(
        &mut self,
        logs: Arc<RwLock<LogStore>>,
        pool_service: Arc<ProviderPoolService>,
        token_cache: Arc<TokenCacheService>,
        db: Option<DbConnection>,
        shared_stats: Option<Arc<parking_lot::RwLock<crate::telemetry::StatsAggregator>>>,
        shared_tokens: Option<Arc<parking_lot::RwLock<crate::telemetry::TokenTracker>>>,
        shared_logger: Option<Arc<crate::telemetry::RequestLogger>>,
        shared_flow_monitor: Option<Arc<FlowMonitor>>,
        shared_flow_interceptor: Option<Arc<FlowInterceptor>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.running {
            return Ok(());
        }

        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        let host = self.config.server.host.clone();
        let port = self.config.server.port;
        let api_key = self.config.server.api_key.clone();
        let api_key_for_state = api_key.clone(); // 用于保存到 running_api_key
        let default_provider_ref = self.default_provider_ref.clone();

        // 重新加载凭证
        let _ = self.kiro_provider.load_credentials().await;
        let kiro = self.kiro_provider.clone();

        // 创建参数注入器
        let injection_enabled = self.config.injection.enabled;
        let injector = Injector::with_rules(
            self.config
                .injection
                .rules
                .iter()
                .map(|r| r.clone().into())
                .collect(),
        );

        // 获取配置和配置路径用于热重载
        let config = self.config.clone();
        let config_path = crate::config::ConfigManager::default_config_path();

        tokio::spawn(async move {
            if let Err(e) = run_server(
                &host,
                port,
                &api_key,
                default_provider_ref,
                kiro,
                logs,
                rx,
                pool_service,
                token_cache,
                db,
                injector,
                injection_enabled,
                shared_stats,
                shared_tokens,
                shared_logger,
                shared_flow_monitor,
                shared_flow_interceptor,
                Some(config),
                Some(config_path),
            )
            .await
            {
                tracing::error!("Server error: {}", e);
            }
        });

        self.running = true;
        self.start_time = Some(std::time::Instant::now());
        // 保存服务器运行时使用的 API key，用于 test_api 命令
        self.running_api_key = Some(api_key_for_state);
        Ok(())
    }

    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        self.running = false;
        self.start_time = None;
        self.running_api_key = None;
    }
}

impl Clone for KiroProvider {
    fn clone(&self) -> Self {
        Self {
            credentials: self.credentials.clone(),
            client: reqwest::Client::new(),
            creds_path: self.creds_path.clone(),
        }
    }
}

pub mod handlers;

#[derive(Clone)]
#[allow(dead_code)]
pub struct AppState {
    pub api_key: String,
    pub base_url: String,
    pub default_provider: Arc<RwLock<String>>,
    pub kiro: Arc<RwLock<KiroProvider>>,
    pub logs: Arc<RwLock<LogStore>>,
    pub kiro_refresh_lock: Arc<tokio::sync::Mutex<()>>,
    pub gemini_refresh_lock: Arc<tokio::sync::Mutex<()>>,
    pub qwen_refresh_lock: Arc<tokio::sync::Mutex<()>>,
    pub pool_service: Arc<ProviderPoolService>,
    pub token_cache: Arc<TokenCacheService>,
    pub db: Option<DbConnection>,
    /// 参数注入器
    pub injector: Arc<RwLock<Injector>>,
    /// 是否启用参数注入
    pub injection_enabled: Arc<RwLock<bool>>,
    /// 请求处理器
    pub processor: Arc<RequestProcessor>,
    /// WebSocket 连接管理器
    pub ws_manager: Arc<WsConnectionManager>,
    /// WebSocket 统计信息
    pub ws_stats: Arc<WsStats>,
    /// 热重载管理器
    pub hot_reload_manager: Option<Arc<HotReloadManager>>,
    /// 请求日志记录器（与 TelemetryState 共享）
    pub request_logger: Option<Arc<crate::telemetry::RequestLogger>>,
    /// Amp CLI 路由器
    pub amp_router: Arc<crate::router::AmpRouter>,
    /// Flow 监控服务
    pub flow_monitor: Arc<FlowMonitor>,
    /// Flow 拦截器
    pub flow_interceptor: Arc<FlowInterceptor>,
    /// 端点 Provider 配置
    pub endpoint_providers: Arc<RwLock<EndpointProvidersConfig>>,
    /// Kiro 事件服务
    pub kiro_event_service: Arc<KiroEventService>,
}

/// 启动配置文件监控
///
/// 监控配置文件变化并触发热重载。
///
/// # 连接保持
///
/// 热重载过程不会中断现有连接：
/// - 配置更新在独立的 tokio 任务中异步执行
/// - 使用 RwLock 进行原子性更新，不会阻塞正在处理的请求
/// - 服务器继续运行，不需要重启
/// - HTTP 和 WebSocket 连接保持活跃
async fn start_config_watcher(
    config_path: PathBuf,
    hot_reload_manager: Option<Arc<HotReloadManager>>,
    processor: Arc<RequestProcessor>,
    logs: Arc<RwLock<LogStore>>,
    db: Option<DbConnection>,
    config_manager: Option<Arc<std::sync::RwLock<ConfigManager>>>,
) -> Option<FileWatcher> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ConfigChangeEvent>();

    // 创建文件监控器
    let mut watcher = match FileWatcher::new(&config_path, tx) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!("[HOT_RELOAD] 创建文件监控器失败: {}", e);
            return None;
        }
    };

    // 启动监控
    if let Err(e) = watcher.start() {
        tracing::error!("[HOT_RELOAD] 启动文件监控失败: {}", e);
        return None;
    }

    tracing::info!("[HOT_RELOAD] 配置文件监控已启动: {:?}", config_path);

    // 启动事件处理任务
    let hot_reload_manager_clone = hot_reload_manager.clone();
    let processor_clone = processor.clone();
    let logs_clone = logs.clone();
    let db_clone = db.clone();
    let config_manager_clone = config_manager.clone();

    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            // 只处理修改事件
            if event.kind != ConfigChangeKind::Modified {
                continue;
            }

            tracing::info!("[HOT_RELOAD] 检测到配置文件变更: {:?}", event.path);
            logs_clone.write().await.add(
                "info",
                &format!("[HOT_RELOAD] 检测到配置文件变更: {:?}", event.path),
            );

            // 执行热重载
            if let Some(ref manager) = hot_reload_manager_clone {
                let result = manager.reload();
                match &result {
                    ReloadResult::Success { .. } => {
                        tracing::info!("[HOT_RELOAD] 配置热重载成功");
                        logs_clone
                            .write()
                            .await
                            .add("info", "[HOT_RELOAD] 配置热重载成功");

                        // 更新处理器中的组件
                        let new_config = manager.config();
                        update_processor_config(&processor_clone, &new_config).await;

                        // 同步凭证池
                        if let (Some(ref db), Some(ref cfg_manager)) =
                            (&db_clone, &config_manager_clone)
                        {
                            match sync_credential_pool_from_config(db, cfg_manager, &logs_clone)
                                .await
                            {
                                Ok(count) => {
                                    tracing::info!(
                                        "[HOT_RELOAD] 凭证池同步完成，共 {} 个凭证",
                                        count
                                    );
                                    logs_clone.write().await.add(
                                        "info",
                                        &format!(
                                            "[HOT_RELOAD] 凭证池同步完成，共 {} 个凭证",
                                            count
                                        ),
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!("[HOT_RELOAD] 凭证池同步失败: {}", e);
                                    logs_clone.write().await.add(
                                        "warn",
                                        &format!("[HOT_RELOAD] 凭证池同步失败: {}", e),
                                    );
                                }
                            }
                        }
                    }
                    ReloadResult::RolledBack { error, .. } => {
                        tracing::warn!("[HOT_RELOAD] 配置热重载失败，已回滚: {}", error);
                        logs_clone.write().await.add(
                            "warn",
                            &format!("[HOT_RELOAD] 配置热重载失败，已回滚: {}", error),
                        );
                    }
                    ReloadResult::Failed {
                        error,
                        rollback_error,
                        ..
                    } => {
                        tracing::error!(
                            "[HOT_RELOAD] 配置热重载失败: {}, 回滚错误: {:?}",
                            error,
                            rollback_error
                        );
                        logs_clone.write().await.add(
                            "error",
                            &format!(
                                "[HOT_RELOAD] 配置热重载失败: {}, 回滚错误: {:?}",
                                error, rollback_error
                            ),
                        );
                    }
                }
            }
        }
    });

    Some(watcher)
}

/// 更新处理器配置
///
/// 当配置热重载成功后，更新 RequestProcessor 中的各个组件。
///
/// # 原子性更新
///
/// 每个组件的更新都是原子性的，使用 RwLock 确保：
/// - 正在处理的请求不会看到部分更新的状态
/// - 更新过程不会阻塞新请求的处理
/// - 现有连接不受影响
async fn update_processor_config(processor: &RequestProcessor, config: &Config) {
    // 更新注入器规则
    {
        let mut injector = processor.injector.write().await;
        injector.clear();
        for rule in &config.injection.rules {
            injector.add_rule(rule.clone().into());
        }
        tracing::debug!(
            "[HOT_RELOAD] 注入器规则已更新: {} 条规则",
            config.injection.rules.len()
        );
    }

    // 更新路由器规则
    {
        let mut router = processor.router.write().await;
        router.clear_rules();

        // 如果配置文件中有路由规则，使用配置文件的规则
        // 否则使用默认规则
        if !config.routing.rules.is_empty() {
            for rule in &config.routing.rules {
                // 解析 provider 字符串为 ProviderType
                if let Ok(provider_type) = rule.provider.parse::<crate::ProviderType>() {
                    router.add_rule(crate::router::RoutingRule {
                        pattern: rule.pattern.clone(),
                        target_provider: provider_type,
                        priority: rule.priority,
                        enabled: true,
                    });
                } else {
                    tracing::warn!("[HOT_RELOAD] 无法解析 provider: {}", rule.provider);
                }
            }
            tracing::debug!(
                "[HOT_RELOAD] 路由规则已更新: {} 条规则（来自配置文件）",
                config.routing.rules.len()
            );
        } else {
            // 使用默认路由规则
            router.add_rule(crate::router::RoutingRule::new(
                "gemini-*",
                crate::ProviderType::Antigravity,
                10,
            ));
            router.add_rule(crate::router::RoutingRule::new(
                "claude-*",
                crate::ProviderType::Kiro,
                10,
            ));
            tracing::debug!(
                "[HOT_RELOAD] 路由规则已更新: 使用默认规则 (gemini-* → Antigravity, claude-* → Kiro)"
            );
        }
    }

    // 更新模型映射器
    {
        let mut mapper = processor.mapper.write().await;
        mapper.clear();
        for (alias, model) in &config.routing.model_aliases {
            mapper.add_alias(alias, model);
        }
        tracing::debug!(
            "[HOT_RELOAD] 模型别名已更新: {} 个别名",
            config.routing.model_aliases.len()
        );
    }

    // 注意：重试配置目前不支持热更新，因为 Retrier 是不可变的
    // 如果需要更新重试配置，需要重启服务器
    tracing::debug!(
        "[HOT_RELOAD] 重试配置: max_retries={}, base_delay={}ms (需重启生效)",
        config.retry.max_retries,
        config.retry.base_delay_ms
    );

    tracing::info!("[HOT_RELOAD] 处理器配置更新完成");
}

/// 从配置同步凭证池
///
/// 当配置热重载成功后，从 YAML 配置中加载凭证并同步到数据库。
///
/// # 同步策略
///
/// - 从配置中加载所有凭证
/// - 对于配置中存在但数据库中不存在的凭证，添加到数据库
/// - 对于配置中存在且数据库中也存在的凭证，更新数据库中的记录
/// - 对于数据库中存在但配置中不存在的凭证，保留（不删除，避免丢失运行时状态）
async fn sync_credential_pool_from_config(
    db: &DbConnection,
    config_manager: &Arc<std::sync::RwLock<ConfigManager>>,
    _logs: &Arc<RwLock<LogStore>>,
) -> Result<usize, String> {
    // 创建凭证同步服务
    let sync_service = CredentialSyncService::new(config_manager.clone());

    // 从配置加载凭证
    let credentials = sync_service.load_from_config().map_err(|e| e.to_string())?;

    let conn = db.lock().map_err(|e| e.to_string())?;
    let mut synced_count = 0;

    for cred in &credentials {
        // 检查凭证是否已存在
        let existing =
            ProviderPoolDao::get_by_uuid(&conn, &cred.uuid).map_err(|e| e.to_string())?;

        if existing.is_some() {
            // 更新现有凭证
            ProviderPoolDao::update(&conn, cred).map_err(|e| e.to_string())?;
            tracing::debug!(
                "[HOT_RELOAD] 更新凭证: {} ({})",
                cred.uuid,
                cred.provider_type
            );
        } else {
            // 添加新凭证
            ProviderPoolDao::insert(&conn, cred).map_err(|e| e.to_string())?;
            tracing::debug!(
                "[HOT_RELOAD] 添加凭证: {} ({})",
                cred.uuid,
                cred.provider_type
            );
        }
        synced_count += 1;
    }

    Ok(synced_count)
}

async fn run_server(
    host: &str,
    port: u16,
    api_key: &str,
    default_provider: Arc<RwLock<String>>,
    kiro: KiroProvider,
    logs: Arc<RwLock<LogStore>>,
    shutdown: oneshot::Receiver<()>,
    pool_service: Arc<ProviderPoolService>,
    token_cache: Arc<TokenCacheService>,
    db: Option<DbConnection>,
    injector: Injector,
    injection_enabled: bool,
    shared_stats: Option<Arc<parking_lot::RwLock<crate::telemetry::StatsAggregator>>>,
    shared_tokens: Option<Arc<parking_lot::RwLock<crate::telemetry::TokenTracker>>>,
    shared_logger: Option<Arc<crate::telemetry::RequestLogger>>,
    shared_flow_monitor: Option<Arc<FlowMonitor>>,
    shared_flow_interceptor: Option<Arc<FlowInterceptor>>,
    config: Option<Config>,
    config_path: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let base_url = format!("http://{}:{}", host, port);

    // 创建请求处理器（使用共享的遥测实例或默认实例）
    let processor = match (shared_stats, shared_tokens) {
        (Some(stats), Some(tokens)) => Arc::new(RequestProcessor::with_shared_telemetry(
            pool_service.clone(),
            stats,
            tokens,
        )),
        _ => Arc::new(RequestProcessor::with_defaults(pool_service.clone())),
    };

    // 将注入器规则同步到处理器
    {
        let mut proc_injector = processor.injector.write().await;
        for rule in injector.rules() {
            proc_injector.add_rule(rule.clone());
        }
    }

    // 初始化 WebSocket 管理器
    let ws_manager = Arc::new(WsConnectionManager::new(WsConfig::default()));
    let ws_stats = ws_manager.stats().clone();

    // 初始化热重载管理器
    let hot_reload_manager = match (&config, &config_path) {
        (Some(cfg), Some(path)) => Some(Arc::new(HotReloadManager::new(cfg.clone(), path.clone()))),
        _ => None,
    };

    // 初始化配置管理器（用于凭证池同步）
    let config_manager: Option<Arc<std::sync::RwLock<ConfigManager>>> =
        match (&config, &config_path) {
            (Some(cfg), Some(path)) => Some(Arc::new(std::sync::RwLock::new(
                ConfigManager::with_config(cfg.clone(), path.clone()),
            ))),
            _ => None,
        };

    let logs_clone = logs.clone();
    let db_clone = db.clone();

    // 初始化 Amp CLI 路由器
    let amp_router = Arc::new(crate::router::AmpRouter::new(
        config
            .as_ref()
            .map(|c| c.ampcode.clone())
            .unwrap_or_default(),
    ));

    // 使用共享的 Flow 监控服务，如果没有则创建新的
    let flow_monitor = shared_flow_monitor
        .unwrap_or_else(|| Arc::new(FlowMonitor::new(FlowMonitorConfig::default(), None)));

    // 使用共享的 Flow 拦截器，如果没有则创建新的
    let flow_interceptor =
        shared_flow_interceptor.unwrap_or_else(|| Arc::new(FlowInterceptor::default()));

    // 初始化端点 Provider 配置
    let endpoint_providers = Arc::new(RwLock::new(
        config
            .as_ref()
            .map(|c| c.endpoint_providers.clone())
            .unwrap_or_default(),
    ));

    // 创建 Kiro 事件服务
    let kiro_event_service = Arc::new(KiroEventService::new());

    let state = AppState {
        api_key: api_key.to_string(),
        base_url,
        default_provider,
        kiro: Arc::new(RwLock::new(kiro)),
        logs,
        kiro_refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        gemini_refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        qwen_refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        pool_service,
        token_cache,
        db,
        injector: Arc::new(RwLock::new(injector)),
        injection_enabled: Arc::new(RwLock::new(injection_enabled)),
        processor: processor.clone(),
        ws_manager,
        ws_stats,
        hot_reload_manager: hot_reload_manager.clone(),
        request_logger: shared_logger,
        amp_router,
        flow_monitor,
        flow_interceptor,
        endpoint_providers,
        kiro_event_service,
    };

    // 启动配置文件监控
    let _file_watcher = if let Some(path) = config_path {
        start_config_watcher(
            path,
            hot_reload_manager,
            processor,
            logs_clone,
            db_clone,
            config_manager,
        )
        .await
    } else {
        None
    };

    // 设置请求体大小限制为 100MB，支持大型上下文请求（如 Claude Code 的 /compact 命令）
    let body_limit = 100 * 1024 * 1024; // 100MB

    // 创建管理 API 路由（带认证中间件）
    let management_config = config
        .as_ref()
        .map(|c| c.remote_management.clone())
        .unwrap_or_default();

    let management_routes = Router::new()
        .route("/v0/management/status", get(handlers::management_status))
        .route(
            "/v0/management/credentials",
            get(handlers::management_list_credentials),
        )
        .route(
            "/v0/management/credentials",
            post(handlers::management_add_credential),
        )
        .route(
            "/v0/management/config",
            get(handlers::management_get_config),
        )
        .route(
            "/v0/management/config",
            axum::routing::put(handlers::management_update_config),
        )
        .layer(crate::middleware::ManagementAuthLayer::new(
            management_config,
        ));

    // Kiro凭证管理API路由
    let kiro_api_routes = Router::new()
        .route(
            "/api/kiro/credentials/available",
            get(handlers::get_available_credentials),
        )
        .route(
            "/api/kiro/credentials/select",
            post(handlers::select_credential),
        )
        .route(
            "/api/kiro/credentials/:uuid/refresh",
            axum::routing::put(handlers::refresh_credential),
        )
        .route(
            "/api/kiro/credentials/:uuid/status",
            get(handlers::get_credential_status),
        );

    // 凭证 API 路由（用于 aster Agent 集成）
    let credentials_api_routes = Router::new()
        .route("/v1/credentials/select", post(handlers::credentials_select))
        .route(
            "/v1/credentials/:uuid/token",
            get(handlers::credentials_get_token),
        );

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/routes", get(list_routes))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/messages", post(handlers::anthropic_messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        // 图像生成 API 路由
        .route(
            "/v1/images/generations",
            post(handlers::handle_image_generation),
        )
        // Gemini 原生协议路由
        .route("/v1/gemini/*path", post(gemini_generate_content))
        // WebSocket 路由
        .route("/v1/ws", get(handlers::ws_upgrade_handler))
        .route("/ws", get(handlers::ws_upgrade_handler))
        // 多供应商路由
        .route(
            "/:selector/v1/messages",
            post(anthropic_messages_with_selector),
        )
        .route(
            "/:selector/v1/chat/completions",
            post(chat_completions_with_selector),
        )
        // Amp CLI 路由
        .route(
            "/api/provider/:provider/v1/chat/completions",
            post(amp_chat_completions),
        )
        .route("/api/provider/:provider/v1/messages", post(amp_messages))
        // Amp CLI 管理代理路由
        .route(
            "/api/auth/*path",
            axum::routing::any(amp_management_proxy_auth),
        )
        .route(
            "/api/user/*path",
            axum::routing::any(amp_management_proxy_user),
        )
        // 管理 API 路由
        .merge(management_routes)
        // Kiro凭证管理API路由
        .merge(kiro_api_routes)
        // 凭证 API 路由（用于 aster Agent 集成）
        .merge(credentials_api_routes)
        .layer(DefaultBodyLimit::max(body_limit))
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("Server listening on {}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = shutdown.await;
        })
        .await?;

    Ok(())
}

async fn count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(_request): Json<serde_json::Value>,
) -> Response {
    if let Err(e) = handlers::verify_api_key(&headers, &state.api_key).await {
        return e.into_response();
    }

    // Claude Code 需要这个端点，返回估算值
    Json(serde_json::json!({
        "input_tokens": 100
    }))
    .into_response()
}

/// Gemini 原生协议处理
/// 路由: POST /v1/gemini/{model}:{method}
/// 例如: /v1/gemini/gemini-3-pro-preview:generateContent
async fn gemini_generate_content(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(path): Path<String>,
    Json(request): Json<serde_json::Value>,
) -> Response {
    if let Err(e) = handlers::verify_api_key(&headers, &state.api_key).await {
        return e.into_response();
    }

    // 解析路径: {model}:{method}
    // 例如: gemini-3-pro-preview:generateContent
    let parts: Vec<&str> = path.splitn(2, ':').collect();
    if parts.len() != 2 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!("无效的路径格式: {}，期望格式: model:method", path)
                }
            })),
        )
            .into_response();
    }

    let model = parts[0];
    let method = parts[1];

    state.logs.write().await.add(
        "info",
        &format!(
            "[GEMINI] POST /v1/gemini/{} model={} method={}",
            path, model, method
        ),
    );

    // 目前只支持 generateContent 方法
    if method != "generateContent" && method != "streamGenerateContent" {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!("不支持的方法: {}，目前只支持 generateContent", method)
                }
            })),
        )
            .into_response();
    }

    let is_stream = method == "streamGenerateContent";

    // 获取默认 provider
    let default_provider = state.default_provider.read().await.clone();

    // 尝试从凭证池中选择 Antigravity 凭证
    let credential = match &state.db {
        Some(db) => state
            .pool_service
            .select_credential(db, &default_provider, Some(model))
            .ok()
            .flatten(),
        None => None,
    };

    let cred = match credential {
        Some(c) => c,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": {
                        "message": "没有可用的 Antigravity 凭证，请先添加凭证"
                    }
                })),
            )
                .into_response();
        }
    };

    state.logs.write().await.add(
        "info",
        &format!(
            "[GEMINI] 使用凭证: type={} name={:?} uuid={}",
            cred.provider_type,
            cred.name,
            &cred.uuid[..8]
        ),
    );

    // 调用 Antigravity Provider
    match &cred.credential {
        CredentialData::AntigravityOAuth {
            creds_file_path,
            project_id,
        } => {
            let mut antigravity = AntigravityProvider::new();
            if let Err(e) = antigravity
                .load_credentials_from_path(creds_file_path)
                .await
            {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!("加载 Antigravity 凭证失败: {}", e)
                        }
                    })),
                )
                    .into_response();
            }

            // 使用新的 validate_token() 方法检查 Token 状态
            let validation_result = antigravity.validate_token();
            tracing::info!(
                "[Antigravity Gemini] Token 验证结果: {:?}",
                validation_result
            );

            // 根据验证结果决定是否刷新
            if validation_result.needs_refresh() {
                tracing::info!("[Antigravity Gemini] Token 需要刷新，开始刷新...");
                match antigravity.refresh_token_with_retry(3).await {
                    Ok(new_token) => {
                        tracing::info!(
                            "[Antigravity Gemini] Token 刷新成功，新 token 长度: {}",
                            new_token.len()
                        );
                    }
                    Err(refresh_error) => {
                        tracing::error!("[Antigravity Gemini] Token 刷新失败: {:?}", refresh_error);

                        // 根据错误类型返回不同的状态码和消息
                        let (status, message) = if refresh_error.requires_reauth() {
                            (StatusCode::UNAUTHORIZED, refresh_error.user_message())
                        } else {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                refresh_error.user_message(),
                            )
                        };

                        return (
                            status,
                            Json(serde_json::json!({
                                "error": {
                                    "message": message
                                }
                            })),
                        )
                            .into_response();
                    }
                }
            }

            // 设置项目 ID
            if let Some(pid) = project_id {
                antigravity.project_id = Some(pid.clone());
            } else if antigravity.project_id.is_none() {
                // 如果凭证中没有 project_id，尝试从 API 获取或生成随机 ID
                if let Err(e) = antigravity.discover_project().await {
                    tracing::warn!("[Antigravity] 获取项目 ID 失败: {}，使用随机生成的 ID", e);
                    // 生成随机项目 ID
                    let uuid = uuid::Uuid::new_v4();
                    let bytes = uuid.as_bytes();
                    let adjectives = ["useful", "bright", "swift", "calm", "bold"];
                    let nouns = ["fuze", "wave", "spark", "flow", "core"];
                    let adj = adjectives[(bytes[0] as usize) % adjectives.len()];
                    let noun = nouns[(bytes[1] as usize) % nouns.len()];
                    let random_part: String = uuid.to_string()[..5].to_lowercase();
                    antigravity.project_id = Some(format!("{}-{}-{}", adj, noun, random_part));
                }
            }

            let proj_id = antigravity.project_id.clone().unwrap_or_else(|| {
                // 最后的后备：生成随机 ID
                let uuid = uuid::Uuid::new_v4();
                format!("proxycast-{}", &uuid.to_string()[..8])
            });

            state
                .logs
                .write()
                .await
                .add("debug", &format!("[GEMINI] 使用 project_id: {}", proj_id));

            // 构建 Antigravity 请求体
            // 直接使用用户传入的 Gemini 格式请求，只添加必要的字段
            let antigravity_request = build_gemini_native_request(&request, model, &proj_id);

            state.logs.write().await.add(
                "debug",
                &format!(
                    "[GEMINI] 请求体: {}",
                    serde_json::to_string(&antigravity_request).unwrap_or_default()
                ),
            );

            if is_stream {
                // 流式响应 - 暂不支持，返回错误
                return (
                    StatusCode::NOT_IMPLEMENTED,
                    Json(serde_json::json!({
                        "error": {
                            "message": "流式响应暂不支持，请使用 generateContent"
                        }
                    })),
                )
                    .into_response();
            }

            // 非流式响应
            match antigravity
                .call_api("generateContent", &antigravity_request)
                .await
            {
                Ok(resp) => {
                    state.logs.write().await.add(
                        "info",
                        &format!(
                            "[GEMINI] 响应成功: {}",
                            serde_json::to_string(&resp)
                                .unwrap_or_default()
                                .chars()
                                .take(200)
                                .collect::<String>()
                        ),
                    );

                    // 直接返回 Gemini 格式响应
                    Json(resp).into_response()
                }
                Err(e) => {
                    state
                        .logs
                        .write()
                        .await
                        .add("error", &format!("[GEMINI] 请求失败: {}", e));

                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": {
                                "message": e.to_string()
                            }
                        })),
                    )
                        .into_response()
                }
            }
        }
        _ => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "Gemini 原生协议只支持 Antigravity 凭证"
                }
            })),
        )
            .into_response(),
    }
}

/// 列出所有可用路由
async fn list_routes(State(state): State<AppState>) -> impl IntoResponse {
    let routes = match &state.db {
        Some(db) => state
            .pool_service
            .get_available_routes(db, &state.base_url)
            .unwrap_or_default(),
        None => Vec::new(),
    };

    // 获取默认 Provider
    let default_provider = state.default_provider.read().await.clone();

    // 添加默认路由
    let mut all_routes = vec![RouteInfo {
        selector: "default".to_string(),
        provider_type: default_provider.clone(),
        credential_count: 1,
        endpoints: vec![
            crate::models::route_model::RouteEndpoint {
                path: "/v1/messages".to_string(),
                protocol: "claude".to_string(),
                url: format!("{}/v1/messages", state.base_url),
            },
            crate::models::route_model::RouteEndpoint {
                path: "/v1/chat/completions".to_string(),
                protocol: "openai".to_string(),
                url: format!("{}/v1/chat/completions", state.base_url),
            },
        ],
        tags: vec!["默认".to_string()],
        enabled: true,
    }];
    all_routes.extend(routes);

    let response = RouteListResponse {
        base_url: state.base_url.clone(),
        default_provider,
        routes: all_routes,
    };

    Json(response)
}

/// 带选择器的 Anthropic messages 处理
async fn anthropic_messages_with_selector(
    State(state): State<AppState>,
    Path(selector): Path<String>,
    headers: HeaderMap,
    Json(request): Json<AnthropicMessagesRequest>,
) -> Response {
    // 使用 Anthropic 格式的认证验证
    if let Err(e) = handlers::verify_api_key_anthropic(&headers, &state.api_key).await {
        state.logs.write().await.add(
            "warn",
            &format!("Unauthorized request to /{}/v1/messages", selector),
        );
        return e.into_response();
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[REQ] POST /{}/v1/messages model={} stream={}",
            selector, request.model, request.stream
        ),
    );

    // 尝试解析凭证
    let credential = match &state.db {
        Some(db) => {
            // 首先尝试按名称查找
            if let Ok(Some(cred)) = state.pool_service.get_by_name(db, &selector) {
                Some(cred)
            }
            // 然后尝试按 UUID 查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_uuid(db, &selector) {
                Some(cred)
            }
            // 最后尝试按 provider 类型轮询
            else if let Ok(Some(cred)) =
                state
                    .pool_service
                    .select_credential(db, &selector, Some(&request.model))
            {
                Some(cred)
            } else {
                None
            }
        }
        None => None,
    };

    match credential {
        Some(cred) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[ROUTE] Using credential: type={} name={:?} uuid={}",
                    cred.provider_type,
                    cred.name,
                    &cred.uuid[..8]
                ),
            );

            // 根据凭证类型调用相应的 Provider
            // 注意：这里没有 Flow 捕获，因为是通过 selector 路由的请求
            handlers::call_provider_anthropic(&state, &cred, &request, None).await
        }
        None => {
            // 回退到默认 Kiro provider
            state.logs.write().await.add(
                "warn",
                &format!(
                    "[ROUTE] Credential not found for selector '{}', falling back to default",
                    selector
                ),
            );
            // 调用原有的 Kiro 处理逻辑
            anthropic_messages_internal(&state, &request).await
        }
    }
}

/// 带选择器的 OpenAI chat completions 处理
async fn chat_completions_with_selector(
    State(state): State<AppState>,
    Path(selector): Path<String>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(e) = handlers::verify_api_key(&headers, &state.api_key).await {
        state.logs.write().await.add(
            "warn",
            &format!("Unauthorized request to /{}/v1/chat/completions", selector),
        );
        return e.into_response();
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[REQ] POST /{}/v1/chat/completions model={} stream={}",
            selector, request.model, request.stream
        ),
    );

    // 尝试解析凭证
    let credential = match &state.db {
        Some(db) => {
            if let Ok(Some(cred)) = state.pool_service.get_by_name(db, &selector) {
                Some(cred)
            } else if let Ok(Some(cred)) = state.pool_service.get_by_uuid(db, &selector) {
                Some(cred)
            } else if let Ok(Some(cred)) =
                state
                    .pool_service
                    .select_credential(db, &selector, Some(&request.model))
            {
                Some(cred)
            } else {
                None
            }
        }
        None => None,
    };

    match credential {
        Some(cred) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[ROUTE] Using credential: type={} name={:?} uuid={}",
                    cred.provider_type,
                    cred.name,
                    &cred.uuid[..8]
                ),
            );

            // 注意：这里没有 Flow 捕获，因为是通过 selector 路由的请求
            handlers::call_provider_openai(&state, &cred, &request, None).await
        }
        None => {
            state.logs.write().await.add(
                "warn",
                &format!(
                    "[ROUTE] Credential not found for selector '{}', falling back to default",
                    selector
                ),
            );
            chat_completions_internal(&state, &request).await
        }
    }
}

// ============ Amp CLI 路由处理 ============

/// Amp CLI chat completions 处理
///
/// 处理 `/api/provider/:provider/v1/chat/completions` 路由
/// 支持模型映射，将不可用模型映射到可用替代
async fn amp_chat_completions(
    State(state): State<AppState>,
    Path(provider): Path<String>,
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(e) = handlers::verify_api_key(&headers, &state.api_key).await {
        state.logs.write().await.add(
            "warn",
            &format!(
                "Unauthorized request to /api/provider/{}/v1/chat/completions",
                provider
            ),
        );
        return e.into_response();
    }

    // 应用模型映射
    let original_model = request.model.clone();
    let mapped_model = state.amp_router.apply_model_mapping(&request.model);
    if mapped_model != original_model {
        state.logs.write().await.add(
            "info",
            &format!(
                "[AMP] Model mapping applied: {} -> {}",
                original_model, mapped_model
            ),
        );
        request.model = mapped_model;
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[AMP] POST /api/provider/{}/v1/chat/completions model={} stream={}",
            provider, request.model, request.stream
        ),
    );

    // 尝试根据 provider 名称选择凭证
    let credential = match &state.db {
        Some(db) => {
            // 首先尝试按 provider 类型选择
            if let Ok(Some(cred)) =
                state
                    .pool_service
                    .select_credential(db, &provider, Some(&request.model))
            {
                Some(cred)
            }
            // 然后尝试按名称查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_name(db, &provider) {
                Some(cred)
            }
            // 最后尝试按 UUID 查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_uuid(db, &provider) {
                Some(cred)
            } else {
                None
            }
        }
        None => None,
    };

    match credential {
        Some(cred) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[AMP] Using credential: type={} name={:?} uuid={}",
                    cred.provider_type,
                    cred.name,
                    &cred.uuid[..8]
                ),
            );
            // 注意：这里没有 Flow 捕获，因为是通过 AMP CLI 路由的请求
            handlers::call_provider_openai(&state, &cred, &request, None).await
        }
        None => {
            state.logs.write().await.add(
                "warn",
                &format!(
                    "[AMP] Credential not found for provider '{}', falling back to default",
                    provider
                ),
            );
            chat_completions_internal(&state, &request).await
        }
    }
}

/// Amp CLI messages 处理
///
/// 处理 `/api/provider/:provider/v1/messages` 路由
/// 支持模型映射，将不可用模型映射到可用替代
async fn amp_messages(
    State(state): State<AppState>,
    Path(provider): Path<String>,
    headers: HeaderMap,
    Json(mut request): Json<AnthropicMessagesRequest>,
) -> Response {
    // 使用 Anthropic 格式的认证验证
    if let Err(e) = handlers::verify_api_key_anthropic(&headers, &state.api_key).await {
        state.logs.write().await.add(
            "warn",
            &format!(
                "Unauthorized request to /api/provider/{}/v1/messages",
                provider
            ),
        );
        return e.into_response();
    }

    // 应用模型映射
    let original_model = request.model.clone();
    let mapped_model = state.amp_router.apply_model_mapping(&request.model);
    if mapped_model != original_model {
        state.logs.write().await.add(
            "info",
            &format!(
                "[AMP] Model mapping applied: {} -> {}",
                original_model, mapped_model
            ),
        );
        request.model = mapped_model;
    }

    state.logs.write().await.add(
        "info",
        &format!(
            "[AMP] POST /api/provider/{}/v1/messages model={} stream={}",
            provider, request.model, request.stream
        ),
    );

    // 尝试根据 provider 名称选择凭证
    let credential = match &state.db {
        Some(db) => {
            // 首先尝试按 provider 类型选择
            if let Ok(Some(cred)) =
                state
                    .pool_service
                    .select_credential(db, &provider, Some(&request.model))
            {
                Some(cred)
            }
            // 然后尝试按名称查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_name(db, &provider) {
                Some(cred)
            }
            // 最后尝试按 UUID 查找
            else if let Ok(Some(cred)) = state.pool_service.get_by_uuid(db, &provider) {
                Some(cred)
            } else {
                None
            }
        }
        None => None,
    };

    match credential {
        Some(cred) => {
            state.logs.write().await.add(
                "info",
                &format!(
                    "[AMP] Using credential: type={} name={:?} uuid={}",
                    cred.provider_type,
                    cred.name,
                    &cred.uuid[..8]
                ),
            );
            // 注意：这里没有 Flow 捕获，因为是通过 AMP CLI 路由的请求
            handlers::call_provider_anthropic(&state, &cred, &request, None).await
        }
        None => {
            state.logs.write().await.add(
                "warn",
                &format!(
                    "[AMP] Credential not found for provider '{}', falling back to default",
                    provider
                ),
            );
            anthropic_messages_internal(&state, &request).await
        }
    }
}

/// Amp CLI 管理代理 - auth 路由
///
/// 处理 `/api/auth/*` 路由，将请求代理到上游 URL
async fn amp_management_proxy_auth(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> Response {
    amp_management_proxy_internal(state, &format!("auth/{}", path), headers, method, body).await
}

/// Amp CLI 管理代理 - user 路由
///
/// 处理 `/api/user/*` 路由，将请求代理到上游 URL
async fn amp_management_proxy_user(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> Response {
    amp_management_proxy_internal(state, &format!("user/{}", path), headers, method, body).await
}

/// Amp CLI 管理代理内部实现
///
/// 处理 `/api/auth/*` 和 `/api/user/*` 路由
/// 将请求代理到上游 URL
///
/// # 参数
/// - `path`: 请求路径（不含 /api/ 前缀，如 "auth/login" 或 "user/profile"）
async fn amp_management_proxy_internal(
    state: AppState,
    path: &str,
    headers: HeaderMap,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> Response {
    let full_path = format!("/api/{}", path);

    // 检查是否是管理路由
    if !state.amp_router.is_management_route(&full_path) {
        state.logs.write().await.add(
            "warn",
            &format!("[AMP] Invalid management route: {}", full_path),
        );
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": {"message": "Not found"}})),
        )
            .into_response();
    }

    // 检查 localhost 限制
    if state.amp_router.restrict_management_to_localhost() {
        // 从 headers 中获取客户端 IP
        let client_ip = headers
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(',').next().unwrap_or("").trim().to_string())
            .or_else(|| {
                headers
                    .get("x-real-ip")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_string())
            });

        if let Some(ip) = &client_ip {
            let is_localhost = ip == "127.0.0.1" || ip == "::1" || ip == "localhost";
            if !is_localhost {
                state.logs.write().await.add(
                    "warn",
                    &format!("[AMP] Management proxy blocked from non-localhost: {}", ip),
                );
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({"error": {"message": "Management endpoints are restricted to localhost"}})),
                )
                    .into_response();
            }
        }
    }

    // 获取上游 URL
    let upstream_url = match state.amp_router.get_management_upstream_path(&full_path) {
        Some(url) => url,
        None => {
            state.logs.write().await.add(
                "warn",
                "[AMP] No upstream URL configured for management proxy",
            );
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": {"message": "Upstream URL not configured"}})),
            )
                .into_response();
        }
    };

    state.logs.write().await.add(
        "info",
        &format!(
            "[AMP] Proxying management request: {} {} -> {}",
            method, full_path, upstream_url
        ),
    );

    // 创建 HTTP 客户端
    let client = reqwest::Client::new();

    // 构建请求
    let mut request_builder = match method {
        axum::http::Method::GET => client.get(&upstream_url),
        axum::http::Method::POST => client.post(&upstream_url),
        axum::http::Method::PUT => client.put(&upstream_url),
        axum::http::Method::DELETE => client.delete(&upstream_url),
        axum::http::Method::PATCH => client.patch(&upstream_url),
        axum::http::Method::HEAD => client.head(&upstream_url),
        axum::http::Method::OPTIONS => client.request(reqwest::Method::OPTIONS, &upstream_url),
        _ => {
            return (
                StatusCode::METHOD_NOT_ALLOWED,
                Json(serde_json::json!({"error": {"message": "Method not allowed"}})),
            )
                .into_response();
        }
    };

    // 复制请求头（排除 host 和 content-length）
    for (name, value) in headers.iter() {
        let name_str = name.as_str().to_lowercase();
        if name_str != "host" && name_str != "content-length" {
            if let Ok(value_str) = value.to_str() {
                request_builder = request_builder.header(name.as_str(), value_str);
            }
        }
    }

    // 添加请求体
    if !body.is_empty() {
        request_builder = request_builder.body(body.to_vec());
    }

    // 发送请求
    match request_builder.send().await {
        Ok(response) => {
            let status = response.status();
            let response_headers = response.headers().clone();

            match response.bytes().await {
                Ok(response_body) => {
                    let mut builder = Response::builder().status(status.as_u16());

                    // 复制响应头
                    for (name, value) in response_headers.iter() {
                        let name_str = name.as_str().to_lowercase();
                        // 排除 transfer-encoding 和 content-length（axum 会自动处理）
                        if name_str != "transfer-encoding" && name_str != "content-length" {
                            builder = builder.header(name.as_str(), value.to_str().unwrap_or(""));
                        }
                    }

                    builder
                        .body(Body::from(response_body.to_vec()))
                        .unwrap_or_else(|_| {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({"error": {"message": "Failed to build response"}})),
                            )
                                .into_response()
                        })
                }
                Err(e) => {
                    state.logs.write().await.add(
                        "error",
                        &format!("[AMP] Failed to read upstream response: {}", e),
                    );
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(serde_json::json!({"error": {"message": format!("Failed to read upstream response: {}", e)}})),
                    )
                        .into_response()
                }
            }
        }
        Err(e) => {
            state.logs.write().await.add(
                "error",
                &format!("[AMP] Failed to proxy request to upstream: {}", e),
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": {"message": format!("Failed to connect to upstream: {}", e)}})),
            )
                .into_response()
        }
    }
}

/// 内部 Anthropic messages 处理 (使用默认 Kiro)
async fn anthropic_messages_internal(
    state: &AppState,
    request: &AnthropicMessagesRequest,
) -> Response {
    // 检查 token
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            if let Err(e) = kiro.refresh_token().await {
                state
                    .logs
                    .write()
                    .await
                    .add("error", &format!("[AUTH] Token refresh failed: {e}"));
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                )
                    .into_response();
            }
        }
    }

    let openai_request = convert_anthropic_to_openai(request);
    let kiro = state.kiro.read().await;

    match kiro.call_api(&openai_request).await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                match resp.bytes().await {
                    Ok(bytes) => {
                        let body = String::from_utf8_lossy(&bytes).to_string();
                        let parsed = parse_cw_response(&body);
                        if request.stream {
                            build_anthropic_stream_response(&request.model, &parsed)
                        } else {
                            build_anthropic_response(&request.model, &parsed)
                        }
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response(),
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}})),
                )
                    .into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string()}})),
        )
            .into_response(),
    }
}

/// 内部 OpenAI chat completions 处理 (使用默认 Kiro)
async fn chat_completions_internal(state: &AppState, request: &ChatCompletionRequest) -> Response {
    {
        let _guard = state.kiro_refresh_lock.lock().await;
        let mut kiro = state.kiro.write().await;
        let needs_refresh =
            kiro.credentials.access_token.is_none() || kiro.is_token_expiring_soon();
        if needs_refresh {
            if let Err(e) = kiro.refresh_token().await {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": {"message": format!("Token refresh failed: {e}")}})),
                )
                    .into_response();
            }
        }
    }

    let kiro = state.kiro.read().await;
    match kiro.call_api(request).await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                match resp.text().await {
                    Ok(body) => {
                        let parsed = parse_cw_response(&body);
                        let has_tool_calls = !parsed.tool_calls.is_empty();

                        let message = if has_tool_calls {
                            serde_json::json!({
                                "role": "assistant",
                                "content": if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(parsed.content) },
                                "tool_calls": parsed.tool_calls.iter().map(|tc| {
                                    serde_json::json!({
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    })
                                }).collect::<Vec<_>>()
                            })
                        } else {
                            serde_json::json!({
                                "role": "assistant",
                                "content": parsed.content
                            })
                        };

                        let response = serde_json::json!({
                            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                            "object": "chat.completion",
                            "created": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "message": message,
                                "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                            }],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        });
                        Json(response).into_response()
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": {"message": e.to_string()}})),
                    )
                        .into_response(),
                }
            } else {
                let body = resp.text().await.unwrap_or_default();
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(serde_json::json!({"error": {"message": format!("Upstream error: {}", body)}})),
                )
                    .into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string()}})),
        )
            .into_response(),
    }
}
