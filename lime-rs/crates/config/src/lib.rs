//! Lime 配置观察者模块
//!
//! 提供基于观察者模式的配置管理系统。
//! 宿主相关的具体实现由 App Server 或 Desktop Host 边界承接。

pub mod observer;

// 重新导出观察者模块的核心类型
pub use observer::emitter::ConfigEventEmit;
pub use observer::events::{
    AmpConfigChangeEvent, ConfigChangeEvent, ConfigChangeSource, EndpointProvidersChangeEvent,
    FullReloadEvent, InjectionChangeEvent, LoggingChangeEvent, NativeAgentChangeEvent,
    RetryChangeEvent, RoutingChangeEvent, ServerChangeEvent,
};
pub use observer::manager::GlobalConfigManager;
pub use observer::observers::{
    DefaultProviderRefObserver, EndpointObserver, InjectorObserver, LoggingObserver, RouterObserver,
};
pub use observer::subject::{ConfigSubject, CONFIG_CHANGED_EVENT, CONFIG_RELOAD_EVENT};
pub use observer::traits::{ConfigObserver, FnObserver, SyncConfigObserver, SyncObserverWrapper};

/// 全局配置管理器共享状态
pub struct GlobalConfigManagerState(pub std::sync::Arc<GlobalConfigManager>);

impl GlobalConfigManagerState {
    pub fn new(manager: GlobalConfigManager) -> Self {
        Self(std::sync::Arc::new(manager))
    }
}

impl std::ops::Deref for GlobalConfigManagerState {
    type Target = std::sync::Arc<GlobalConfigManager>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
