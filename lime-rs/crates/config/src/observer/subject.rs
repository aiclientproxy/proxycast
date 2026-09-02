//! 配置主题（Subject）实现
//!
//! 管理配置观察者的注册、注销和通知

use super::emitter::ConfigEventEmit;
use super::events::{ConfigChangeEvent, ConfigChangeSource, FullReloadEvent};
use super::traits::ConfigObserver;
use lime_core::config::Config;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

/// 配置事件名称常量
pub const CONFIG_CHANGED_EVENT: &str = "config-changed";
/// 预留：配置重新加载事件
#[allow(dead_code)]
pub const CONFIG_RELOAD_EVENT: &str = "config-reload";

/// 观察者条目
struct ObserverEntry {
    observer: Arc<dyn ConfigObserver>,
}

/// 配置主题（Subject）
///
/// 管理配置观察者的注册、注销和通知
pub struct ConfigSubject {
    /// 观察者列表（按优先级排序）
    observers: RwLock<BTreeMap<i32, Vec<ObserverEntry>>>,
    /// 当前配置
    current_config: RwLock<Config>,
    /// 事件广播通道
    event_tx: broadcast::Sender<ConfigChangeEvent>,
    /// 事件发射器（抽象宿主事件通道）
    emitter: RwLock<Option<Arc<dyn ConfigEventEmit>>>,
    /// 是否启用事件发射
    events_enabled: RwLock<bool>,
}

impl ConfigSubject {
    /// 创建新的配置主题
    pub fn new(initial_config: Config) -> Self {
        let (event_tx, _) = broadcast::channel(100);

        Self {
            observers: RwLock::new(BTreeMap::new()),
            current_config: RwLock::new(initial_config),
            event_tx,
            emitter: RwLock::new(None),
            events_enabled: RwLock::new(true),
        }
    }

    /// 设置事件发射器
    pub fn set_emitter(&self, emitter: Arc<dyn ConfigEventEmit>) {
        let mut e = self.emitter.write();
        *e = Some(emitter);
        tracing::debug!("[ConfigSubject] 事件发射器已设置");
    }

    /// 启用/禁用事件发射
    pub fn set_events_enabled(&self, enabled: bool) {
        let mut flag = self.events_enabled.write();
        *flag = enabled;
    }

    /// 注册观察者
    pub fn register(&self, observer: Arc<dyn ConfigObserver>) {
        let priority = observer.priority();
        let name = observer.name().to_string();
        let entry = ObserverEntry { observer };

        let mut observers = self.observers.write();
        observers.entry(priority).or_default().push(entry);

        tracing::info!(
            "[ConfigSubject] 注册观察者: {} (优先级: {})",
            name,
            priority
        );
    }

    /// 注销观察者（按名称）
    pub fn unregister(&self, name: &str) {
        let mut observers = self.observers.write();
        for entries in observers.values_mut() {
            entries.retain(|e| e.observer.name() != name);
        }
        observers.retain(|_, v| !v.is_empty());
        tracing::info!("[ConfigSubject] 注销观察者: {}", name);
    }

    /// 获取当前配置（克隆）
    pub fn config(&self) -> Config {
        self.current_config.read().clone()
    }

    /// 获取配置引用
    pub fn config_ref(&self) -> parking_lot::RwLockReadGuard<'_, Config> {
        self.current_config.read()
    }

    /// 直接更新配置（不触发通知，用于内部同步）
    pub fn set_config(&self, config: Config) {
        let mut current = self.current_config.write();
        *current = config;
    }

    /// 更新配置并通知观察者
    pub async fn update_config(&self, new_config: Config, source: ConfigChangeSource) {
        let event = ConfigChangeEvent::FullReload(FullReloadEvent {
            timestamp_ms: Self::current_timestamp_ms(),
            source,
        });

        {
            let mut config = self.current_config.write();
            *config = new_config.clone();
        }

        self.notify_observers(&event, &new_config).await;
        self.emit_event(&event);
        let _ = self.event_tx.send(event);
    }

    /// 通知特定事件（不更新配置）
    pub async fn notify_event(&self, event: ConfigChangeEvent) {
        let config = self.config();
        self.notify_observers(&event, &config).await;
        self.emit_event(&event);
        let _ = self.event_tx.send(event);
    }

    /// 订阅事件广播
    pub fn subscribe(&self) -> broadcast::Receiver<ConfigChangeEvent> {
        self.event_tx.subscribe()
    }

    /// 通知所有观察者
    async fn notify_observers(&self, event: &ConfigChangeEvent, config: &Config) {
        let observers: Vec<Arc<dyn ConfigObserver>> = {
            let observers = self.observers.read();
            observers
                .values()
                .flatten()
                .filter(|e| e.observer.is_interested_in(event))
                .map(|e| e.observer.clone())
                .collect()
        };

        tracing::debug!(
            "[ConfigSubject] 通知 {} 个观察者，事件类型: {}",
            observers.len(),
            event.event_type()
        );

        for observer in observers {
            let name = observer.name().to_string();
            match observer.on_config_changed(event, config).await {
                Ok(()) => {
                    tracing::debug!("[ConfigSubject] 观察者 {} 处理成功", name);
                }
                Err(e) => {
                    tracing::error!("[ConfigSubject] 观察者 {} 处理失败: {}", name, e);
                }
            }
        }
    }

    /// 发送事件（通过抽象发射器）
    fn emit_event(&self, event: &ConfigChangeEvent) {
        let enabled = *self.events_enabled.read();
        if !enabled {
            return;
        }

        let emitter = self.emitter.read();
        if let Some(emitter) = emitter.as_ref() {
            if let Err(e) = emitter.emit_config_event(CONFIG_CHANGED_EVENT, event) {
                tracing::error!("[ConfigSubject] 发送事件失败: {}", e);
            } else {
                tracing::debug!("[ConfigSubject] 已发送事件: {}", event.event_type());
            }
        }
    }

    /// 获取当前时间戳（毫秒）
    fn current_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// 获取观察者数量
    pub fn observer_count(&self) -> usize {
        let observers = self.observers.read();
        observers.values().map(|v| v.len()).sum()
    }

    /// 获取所有观察者名称
    pub fn observer_names(&self) -> Vec<String> {
        let observers = self.observers.read();
        observers
            .values()
            .flatten()
            .map(|e| e.observer.name().to_string())
            .collect()
    }
}

impl Default for ConfigSubject {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::events::ConfigChangeSource;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingObserver {
        name: String,
        priority: i32,
        count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ConfigObserver for CountingObserver {
        fn name(&self) -> &str {
            &self.name
        }

        async fn on_config_changed(
            &self,
            _event: &ConfigChangeEvent,
            _config: &Config,
        ) -> Result<(), String> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn priority(&self) -> i32 {
            self.priority
        }
    }

    #[tokio::test]
    async fn test_register_and_notify() {
        let subject = ConfigSubject::new(Config::default());
        let count = Arc::new(AtomicUsize::new(0));

        let observer = Arc::new(CountingObserver {
            name: "test".to_string(),
            priority: 100,
            count: count.clone(),
        });

        subject.register(observer);
        assert_eq!(subject.observer_count(), 1);

        subject
            .update_config(Config::default(), ConfigChangeSource::ApiCall)
            .await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_unregister() {
        let subject = ConfigSubject::new(Config::default());
        let count = Arc::new(AtomicUsize::new(0));

        let observer = Arc::new(CountingObserver {
            name: "test".to_string(),
            priority: 100,
            count: count.clone(),
        });

        subject.register(observer);
        subject.unregister("test");
        assert_eq!(subject.observer_count(), 0);

        subject
            .update_config(Config::default(), ConfigChangeSource::ApiCall)
            .await;
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_priority_order() {
        let subject = ConfigSubject::new(Config::default());

        for (name, priority) in [("low", 100), ("high", 10), ("medium", 50)] {
            let observer = Arc::new(CountingObserver {
                name: name.to_string(),
                priority,
                count: Arc::new(AtomicUsize::new(0)),
            });
            subject.register(observer);
        }

        let names = subject.observer_names();
        assert_eq!(names.len(), 3);
    }
}
