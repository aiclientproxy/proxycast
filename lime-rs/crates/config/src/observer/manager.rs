//! 全局配置管理器
//!
//! 整合配置主题、热重载和观察者管理。
//! 处理器观察者和宿主观察者由 App Server 边界注册。

use super::emitter::ConfigEventEmit;
use super::events::ConfigChangeSource;
use super::observers::{
    DefaultProviderRefObserver, EndpointObserver, InjectorObserver, LoggingObserver, RouterObserver,
};
use super::subject::ConfigSubject;
use super::traits::ConfigObserver;
use lime_core::config::{Config, EndpointProvidersConfig, HotReloadManager, ReloadResult};
use lime_core::router::{ModelMapper, Router};
use lime_infra::Injector;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 全局配置管理器
pub struct GlobalConfigManager {
    /// 配置主题
    subject: Arc<ConfigSubject>,
    /// 热重载管理器
    hot_reload: Arc<parking_lot::RwLock<HotReloadManager>>,
    /// 配置文件路径
    config_path: PathBuf,
}

impl GlobalConfigManager {
    /// 创建新的全局配置管理器
    pub fn new(config: Config, config_path: PathBuf) -> Self {
        let subject = Arc::new(ConfigSubject::new(config.clone()));
        let hot_reload = Arc::new(parking_lot::RwLock::new(HotReloadManager::new(
            config,
            config_path.clone(),
        )));

        Self {
            subject,
            hot_reload,
            config_path,
        }
    }

    /// 获取配置主题
    pub fn subject(&self) -> Arc<ConfigSubject> {
        self.subject.clone()
    }

    /// 获取当前配置
    pub fn config(&self) -> Config {
        self.subject.config()
    }

    /// 设置事件发射器（替代原来的 set_app_handle）
    pub fn set_emitter(&self, emitter: Arc<dyn ConfigEventEmit>) {
        self.subject.set_emitter(emitter);
    }

    /// 注册观察者
    pub fn register_observer(&self, observer: Arc<dyn ConfigObserver>) {
        self.subject.register(observer);
    }

    /// 注销观察者
    pub fn unregister_observer(&self, name: &str) {
        self.subject.unregister(name);
    }

    /// 注册路由器相关观察者
    pub fn register_router_observers(
        &self,
        router: Arc<RwLock<Router>>,
        mapper: Arc<RwLock<ModelMapper>>,
        injector: Arc<RwLock<Injector>>,
    ) {
        let router_observer = Arc::new(RouterObserver::new(router, mapper));
        self.subject.register(router_observer);

        let injector_observer = Arc::new(InjectorObserver::new(injector));
        self.subject.register(injector_observer);

        let logging_observer = Arc::new(LoggingObserver);
        self.subject.register(logging_observer);

        tracing::info!("[GlobalConfigManager] 已注册路由器相关观察者");
    }

    /// 注册端点 Provider 观察者
    pub fn register_endpoint_observer(
        &self,
        endpoint_providers: Arc<RwLock<EndpointProvidersConfig>>,
    ) {
        let observer = Arc::new(EndpointObserver::new(endpoint_providers));
        self.subject.register(observer);
    }

    /// 注册默认 Provider 引用观察者
    pub fn register_default_provider_ref_observer(
        &self,
        default_provider_ref: Arc<RwLock<String>>,
    ) {
        let observer = Arc::new(DefaultProviderRefObserver::new(default_provider_ref));
        self.subject.register(observer);
    }

    /// 更新配置并通知观察者
    pub async fn update_config(&self, new_config: Config, source: ConfigChangeSource) {
        lime_core::tool_calling::apply_tool_calling_runtime_config(&new_config);
        {
            let hot_reload = self.hot_reload.read();
            hot_reload.update_config(new_config.clone());
        }
        self.subject.update_config(new_config, source).await;
    }

    /// 执行热重载
    pub async fn reload(&self) -> ReloadResult {
        let result = {
            let hot_reload = self.hot_reload.read();
            hot_reload.reload()
        };

        match &result {
            ReloadResult::Success { .. } => {
                let new_config = {
                    let hot_reload = self.hot_reload.read();
                    hot_reload.config()
                };

                self.subject
                    .update_config(new_config, ConfigChangeSource::HotReload)
                    .await;
                tracing::info!("[GlobalConfigManager] 热重载成功");
            }
            ReloadResult::RolledBack { error, .. } => {
                tracing::warn!("[GlobalConfigManager] 热重载失败，已回滚: {}", error);
            }
            ReloadResult::Failed { error, .. } => {
                tracing::error!("[GlobalConfigManager] 热重载失败: {}", error);
            }
        }

        result
    }

    /// 保存配置到文件并通知观察者
    pub async fn save_config(&self, config: &Config) -> Result<(), String> {
        lime_core::config::save_config(config).map_err(|e| e.to_string())?;
        self.update_config(config.clone(), ConfigChangeSource::ApiCall)
            .await;
        Ok(())
    }

    /// 订阅配置变更事件
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<super::events::ConfigChangeEvent> {
        self.subject.subscribe()
    }

    /// 获取配置文件路径
    pub fn config_path(&self) -> &PathBuf {
        &self.config_path
    }

    /// 获取观察者数量
    pub fn observer_count(&self) -> usize {
        self.subject.observer_count()
    }

    /// 获取观察者名称列表
    pub fn observer_names(&self) -> Vec<String> {
        self.subject.observer_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_global_config_manager_creation() {
        let config = Config::default();
        let path = PathBuf::from("/tmp/test_config.yaml");
        let manager = GlobalConfigManager::new(config.clone(), path);

        assert_eq!(manager.observer_count(), 0);
        assert_eq!(manager.config().default_provider, config.default_provider);
    }

    #[tokio::test]
    async fn test_register_observer() {
        let config = Config::default();
        let path = PathBuf::from("/tmp/test_config.yaml");
        let manager = GlobalConfigManager::new(config, path);

        let observer = Arc::new(LoggingObserver);
        manager.register_observer(observer);

        assert_eq!(manager.observer_count(), 1);
        assert!(manager
            .observer_names()
            .contains(&"LoggingObserver".to_string()));
    }
}
