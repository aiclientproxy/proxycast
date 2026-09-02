//! 录音服务桥接层
//!
//! 录音核心逻辑已迁移到 `voice-core` 的 `threaded_recorder` 模块。
//! 本模块提供共享状态包装与稳定导出路径。

use parking_lot::Mutex;
use std::sync::Arc;

pub use voice_core::{AudioDeviceInfo, RecordingCommand, RecordingResponse, RecordingService};

/// 获取所有可用的麦克风设备
pub fn list_audio_devices() -> Result<Vec<AudioDeviceInfo>, String> {
    voice_core::list_audio_devices().map_err(|e| e.to_string())
}

/// 全局录音服务共享状态
pub struct RecordingServiceState(pub Arc<Mutex<RecordingService>>);

impl RecordingServiceState {
    /// 创建新的录音服务状态
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(RecordingService::new())))
    }
}

impl Default for RecordingServiceState {
    fn default() -> Self {
        Self::new()
    }
}

/// 创建录音服务状态
pub fn create_recording_service_state() -> RecordingServiceState {
    RecordingServiceState::new()
}
