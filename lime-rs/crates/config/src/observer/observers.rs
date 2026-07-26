//! 内置配置观察者实现
//!
//! 提供常用组件的配置观察者。
//! TauriObserver 保留在主 crate（依赖 Tauri）。

use super::events::ConfigChangeEvent;
use super::traits::ConfigObserver;
use async_trait::async_trait;
use lime_core::config::{Config, EndpointProvidersConfig};
use lime_core::router::{ModelMapper, Router};
use lime_infra::Injector;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 路由器观察者
///
/// 监听路由配置变更，更新 Router 和 ModelMapper
pub struct RouterObserver {
    router: Arc<RwLock<Router>>,
    mapper: Arc<RwLock<ModelMapper>>,
}

impl RouterObserver {
    pub fn new(router: Arc<RwLock<Router>>, mapper: Arc<RwLock<ModelMapper>>) -> Self {
        Self { router, mapper }
    }
}

#[async_trait]
impl ConfigObserver for RouterObserver {
    fn name(&self) -> &str {
        "RouterObserver"
    }

    fn priority(&self) -> i32 {
        10
    }

    fn is_interested_in(&self, event: &ConfigChangeEvent) -> bool {
        matches!(
            event,
            ConfigChangeEvent::FullReload(_) | ConfigChangeEvent::RoutingChanged(_)
        )
    }

    async fn on_config_changed(
        &self,
        _event: &ConfigChangeEvent,
        config: &Config,
    ) -> Result<(), String> {
        // 更新默认 Provider
        if let Ok(provider_type) = config.default_provider.parse::<lime_core::ProviderType>() {
            let mut router = self.router.write().await;
            router.set_default_provider(provider_type);
            tracing::info!(
                "[RouterObserver] 更新默认 Provider: {}",
                config.default_provider
            );
        }

        // 更新模型别名
        {
            let mut mapper = self.mapper.write().await;
            mapper.clear();
            for (alias, model) in &config.routing.model_aliases {
                mapper.add_alias(alias, model);
            }
            tracing::debug!(
                "[RouterObserver] 更新模型别名: {} 个",
                config.routing.model_aliases.len()
            );
        }

        Ok(())
    }
}

/// 注入器观察者
pub struct InjectorObserver {
    injector: Arc<RwLock<Injector>>,
}

impl InjectorObserver {
    pub fn new(injector: Arc<RwLock<Injector>>) -> Self {
        Self { injector }
    }
}

#[async_trait]
impl ConfigObserver for InjectorObserver {
    fn name(&self) -> &str {
        "InjectorObserver"
    }

    fn priority(&self) -> i32 {
        20
    }

    fn is_interested_in(&self, event: &ConfigChangeEvent) -> bool {
        matches!(
            event,
            ConfigChangeEvent::FullReload(_) | ConfigChangeEvent::InjectionChanged(_)
        )
    }

    async fn on_config_changed(
        &self,
        _event: &ConfigChangeEvent,
        config: &Config,
    ) -> Result<(), String> {
        let mut injector = self.injector.write().await;
        injector.clear();

        for rule in &config.injection.rules {
            injector.add_rule(rule.clone().into());
        }

        tracing::info!(
            "[InjectorObserver] 更新注入规则: {} 条",
            config.injection.rules.len()
        );

        Ok(())
    }
}

/// 端点 Provider 观察者
pub struct EndpointObserver {
    endpoint_providers: Arc<RwLock<EndpointProvidersConfig>>,
}

impl EndpointObserver {
    pub fn new(endpoint_providers: Arc<RwLock<EndpointProvidersConfig>>) -> Self {
        Self { endpoint_providers }
    }
}

#[async_trait]
impl ConfigObserver for EndpointObserver {
    fn name(&self) -> &str {
        "EndpointObserver"
    }

    fn priority(&self) -> i32 {
        30
    }

    fn is_interested_in(&self, event: &ConfigChangeEvent) -> bool {
        matches!(
            event,
            ConfigChangeEvent::FullReload(_) | ConfigChangeEvent::EndpointProvidersChanged(_)
        )
    }

    async fn on_config_changed(
        &self,
        _event: &ConfigChangeEvent,
        config: &Config,
    ) -> Result<(), String> {
        let mut ep = self.endpoint_providers.write().await;
        *ep = config.endpoint_providers.clone();
        tracing::info!("[EndpointObserver] 更新端点 Provider 配置");
        Ok(())
    }
}

/// 日志观察者
pub struct LoggingObserver;

#[async_trait]
impl ConfigObserver for LoggingObserver {
    fn name(&self) -> &str {
        "LoggingObserver"
    }

    fn priority(&self) -> i32 {
        50
    }

    fn is_interested_in(&self, event: &ConfigChangeEvent) -> bool {
        matches!(
            event,
            ConfigChangeEvent::FullReload(_) | ConfigChangeEvent::LoggingChanged(_)
        )
    }

    async fn on_config_changed(
        &self,
        _event: &ConfigChangeEvent,
        config: &Config,
    ) -> Result<(), String> {
        tracing::info!(
            "[LoggingObserver] 日志配置更新: enabled={}, level={}",
            config.logging.enabled,
            config.logging.level
        );
        Ok(())
    }
}

/// 默认 Provider 引用观察者
pub struct DefaultProviderRefObserver {
    default_provider_ref: Arc<RwLock<String>>,
}

impl DefaultProviderRefObserver {
    pub fn new(default_provider_ref: Arc<RwLock<String>>) -> Self {
        Self {
            default_provider_ref,
        }
    }
}

#[async_trait]
impl ConfigObserver for DefaultProviderRefObserver {
    fn name(&self) -> &str {
        "DefaultProviderRefObserver"
    }

    fn priority(&self) -> i32 {
        5
    }

    fn is_interested_in(&self, event: &ConfigChangeEvent) -> bool {
        matches!(
            event,
            ConfigChangeEvent::FullReload(_) | ConfigChangeEvent::RoutingChanged(_)
        )
    }

    async fn on_config_changed(
        &self,
        _event: &ConfigChangeEvent,
        config: &Config,
    ) -> Result<(), String> {
        let mut dp = self.default_provider_ref.write().await;
        *dp = config.default_provider.clone();
        tracing::debug!(
            "[DefaultProviderRefObserver] 更新 default_provider_ref: {}",
            config.default_provider
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::events::{ConfigChangeSource, FullReloadEvent};

    #[tokio::test]
    async fn test_router_observer_priority() {
        let router = Arc::new(RwLock::new(Router::default()));
        let mapper = Arc::new(RwLock::new(ModelMapper::new()));
        let observer = RouterObserver::new(router, mapper);

        assert_eq!(observer.name(), "RouterObserver");
        assert_eq!(observer.priority(), 10);
    }

    #[tokio::test]
    async fn test_injector_observer_interest() {
        let injector = Arc::new(RwLock::new(Injector::new()));
        let observer = InjectorObserver::new(injector);

        let full_reload = ConfigChangeEvent::FullReload(FullReloadEvent {
            timestamp_ms: 0,
            source: ConfigChangeSource::ApiCall,
        });

        assert!(observer.is_interested_in(&full_reload));
    }
}
