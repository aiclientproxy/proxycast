//! 通用事件发射 trait
//!
//! 提供与 Tauri 解耦的事件发射抽象，供独立 crate 使用。
//! 主 crate 通过实现 `EventEmit` trait 注入 Tauri 的 `AppHandle.emit()`。

use std::sync::Arc;

/// 事件发射 trait
///
/// 所有需要向前端发送事件的 crate 都通过此 trait 抽象，
/// 避免直接依赖 Tauri。
pub trait EventEmit: Send + Sync + 'static {
    /// 发送事件
    ///
    /// # Arguments
    /// * `event` - 事件名称（如 "mcp:tools_updated"）
    /// * `payload` - JSON 格式的事件数据
    fn emit_event(&self, event: &str, payload: &serde_json::Value) -> Result<(), String>;
}

/// 动态事件发射器包装
///
/// 使用 `Arc<dyn EventEmit>` 实现可克隆的动态分发。
#[derive(Clone)]
pub struct DynEmitter(pub Arc<dyn EventEmit>);

impl DynEmitter {
    /// 创建新的动态发射器
    pub fn new(emitter: impl EventEmit) -> Self {
        Self(Arc::new(emitter))
    }

    /// 发送事件
    pub fn emit_event(&self, event: &str, payload: &serde_json::Value) -> Result<(), String> {
        self.0.emit_event(event, payload)
    }
}

/// 空操作发射器（用于测试或不需要事件的场景）
#[derive(Debug, Clone)]
pub struct NoOpEmitter;

impl EventEmit for NoOpEmitter {
    fn emit_event(&self, _event: &str, _payload: &serde_json::Value) -> Result<(), String> {
        Ok(())
    }
}
